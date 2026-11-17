"""External specialized tools for metadata extraction (paper Sec. III-A, Figure 2).

These are the concrete tool backends the ReAct processors call:
  - YOLOTool        object detection (ultralytics YOLOv8)
  - Detectron2Tool  object detection (facebookresearch Detectron2)
  - MMDetectionTool object detection (OpenMMLab MMDetection)
  - Places365Tool   scene classification (ResNet-18 trained on Places365)
  - TimmClassifyTool scene/content classification (timm ViT, ImageNet labels)
  - ViTFeatureTool  semantic feature extraction (timm ViT embedding)
  - OpenCVTool      image decoding and basic visual properties (OpenCV)
  - SpacyNERTool    named-entity recognition (spaCy en_core_web_sm)
  - TransformersNERTool named-entity recognition (HF Transformers BERT NER)
  - GLiNERNERTool   named-entity recognition over open types (GLiNER)
  - VaderSentimentTool sentiment (VADER), KeyBERTTool topics (KeyBERT)

Every tool lazy-loads its model on first use; `.available` reports whether the
dependency + weights are actually present, so a processor can honestly record which
tools ran (per the integrity rules — a tool that could not load is reported as
unavailable, never faked). All tools operate on REAL inputs (image bytes/paths/URLs,
raw text).

Further tools are registered before query processing with ``register_tool``
(paper Sec. III-A); the text and image processors offer them to the ReAct agent
next to the built-in ones and merge their output by the kind they declare.
"""
from __future__ import annotations
import io
import os
import threading
import urllib.request
from typing import Any, Callable, Dict, List, Optional

_MODELS_DIR = os.path.expanduser("~/.timm_metadata_models")

# Loaded models, shared by all tool instances and threads (each model is
# loaded once, under a lock, on first use).
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _cached_model(key: str, loader: Callable[[], Any]) -> Any:
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = loader()
        return _MODEL_CACHE[key]


def _models_dir() -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return _MODELS_DIR


def center_saliency(box: List[float], width: float, height: float) -> float:
    """Visual saliency s_o of a detection under the center-bias prior: 1 at
    the image center, decreasing linearly to 0 at the corners."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    dx, dy = (cx - width / 2.0) / max(width / 2.0, 1.0), (cy - height / 2.0) / max(height / 2.0, 1.0)
    return round(max(0.0, 1.0 - ((dx * dx + dy * dy) / 2.0) ** 0.5), 4)


# ---------------------------------------------------------------- image loading
def load_image(src: Any):
    """Load a PIL RGB image from a PIL image, raw bytes, a {bytes} struct, a
    local path, or an http(s) URL."""
    from PIL import Image
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, dict) and src.get("bytes"):
        src = src["bytes"]
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    s = str(src)
    if s.startswith(("http://", "https://")):
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(s).convert("RGB")


def image_bytes(src: Any) -> bytes:
    """The encoded bytes of an image given as raw bytes, a {bytes} struct, a
    local path, an http(s) URL, or a PIL image (re-encoded as PNG)."""
    if isinstance(src, dict) and src.get("bytes"):
        src = src["bytes"]
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    try:
        from PIL import Image
        if isinstance(src, Image.Image):
            buf = io.BytesIO()
            src.save(buf, format="PNG")
            return buf.getvalue()
    except ImportError:
        pass
    s = str(src)
    if s.startswith(("http://", "https://")):
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
    with open(s, "rb") as f:
        return f.read()


class _Tool:
    name = "tool"
    deps: List[str] = []       # modules that must import for the tool to be usable
    _err: Optional[str] = None
    # set on tools added with register_tool: the modality they read and the
    # kind of metadata they return (see register_tool)
    modality: Optional[str] = None
    kind: Optional[str] = None

    @property
    def available(self) -> bool:
        """CHEAP check: only verify the required libraries import — never download
        weights or load a model here (that is _load's job, done lazily on first run)."""
        import importlib
        try:
            for m in self.deps:
                importlib.import_module(m)
            return True
        except Exception as e:  # dependency missing / broken install
            self._err = f"{type(e).__name__}: {str(e)[:90]}"
            return False

    def _load(self):
        raise NotImplementedError


# ---------------------------------------------------------------- YOLO (objects)
class YOLOTool(_Tool):
    name = "detect_objects_yolo"
    deps = ["ultralytics"]
    description = "Detect objects in an image (label, confidence, bbox) via YOLOv8."

    def _load(self):
        def load():
            from ultralytics import YOLO
            local = os.path.join(_models_dir(), "yolov8n.pt")
            return YOLO(local if os.path.exists(local) else "yolov8n.pt")
        return _cached_model("yolo", load)

    def run(self, image_src: Any) -> List[Dict[str, Any]]:
        model = self._load()
        img = load_image(image_src)
        W, H = img.size
        res = model.predict(img, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, W * H)
            out.append({"label": model.names[int(b.cls[0])], "confidence": float(b.conf[0]),
                        "bbox": [x1, y1, x2, y2], "area": round(area, 4),
                        "saliency": center_saliency([x1, y1, x2, y2], W, H)})
        return out


# ---------------------------------------------------------------- Detectron2 (objects)
class Detectron2Tool(_Tool):
    name = "detect_objects_detectron2"
    deps = ["detectron2", "omegaconf"]
    description = "Detect objects in an image via Detectron2 (Mask R-CNN)."

    def _load(self):
        return _cached_model("detectron2", self._build)

    @staticmethod
    def _build():
        import detectron2  # noqa
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        from detectron2.data import MetadataCatalog
        return DefaultPredictor(cfg), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    def run(self, image_src: Any) -> List[Dict[str, Any]]:
        import numpy as np
        predictor, classes = self._load()
        img = load_image(image_src)
        W, H = img.size
        arr = np.array(img)[:, :, ::-1]  # RGB->BGR
        inst = predictor(arr)["instances"].to("cpu")
        out = []
        for box, cls, sc in zip(inst.pred_boxes.tensor.tolist(),
                                inst.pred_classes.tolist(), inst.scores.tolist()):
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, W * H)
            out.append({"label": classes[cls], "confidence": float(sc),
                        "bbox": box, "area": round(area, 4),
                        "saliency": center_saliency(box, W, H)})
        return out


# ---------------------------------------------------------------- MMDetection (objects)
class MMDetectionTool(_Tool):
    """MMDetection (Chen et al., 2019) detector; optional like Detectron2: the
    tool is unavailable, not an error, when ``mmdet`` is not installed."""
    name = "detect_objects_mmdetection"
    deps = ["mmdet", "mmcv"]
    description = "Detect objects in an image via MMDetection (RTMDet)."
    model_name = "rtmdet_tiny_8xb32-300e_coco"
    score_threshold = 0.5

    def _load(self):
        def build():
            from mmdet.apis import DetInferencer
            return DetInferencer(model=self.model_name, device="cpu", show_progress=False)
        return _cached_model("mmdetection", build)

    def run(self, image_src: Any) -> List[Dict[str, Any]]:
        import numpy as np
        inferencer = self._load()
        classes = inferencer.model.dataset_meta["classes"]
        img = load_image(image_src)
        W, H = img.size
        pred = inferencer(np.array(img)[:, :, ::-1], return_vis=False)["predictions"][0]
        out = []
        for label, score, box in zip(pred["labels"], pred["scores"], pred["bboxes"]):
            if float(score) < self.score_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, W * H)
            out.append({"label": classes[int(label)], "confidence": float(score),
                        "bbox": [x1, y1, x2, y2], "area": round(area, 4),
                        "saliency": center_saliency([x1, y1, x2, y2], W, H)})
        return out


# ---------------------------------------------------------------- Places365 (scene)
_PLACES_WEIGHTS = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
_PLACES_CATS = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"


class Places365Tool(_Tool):
    name = "classify_scene_places365"
    deps = ["torch", "torchvision"]
    description = "Classify the scene/place category of an image via ResNet18-Places365."

    def _fetch(self, url, fn):
        path = os.path.join(_models_dir(), fn)
        if not os.path.exists(path):
            # bounded download into a temporary file, renamed only when complete
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, path)
        return path

    def _load(self):
        return _cached_model("places365", self._build)

    def _build(self):
        import torch
        from torchvision.models import resnet18
        wpath = self._fetch(_PLACES_WEIGHTS, "resnet18_places365.pth.tar")
        cpath = self._fetch(_PLACES_CATS, "categories_places365.txt")
        model = resnet18(num_classes=365)
        ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
        model.eval()
        cats = [ln.strip().split(" ")[0][3:] for ln in open(cpath)]
        return model, cats

    def run(self, image_src: Any, topk: int = 3) -> List[Dict[str, Any]]:
        import torch
        from torchvision import transforms
        model, cats = self._load()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
        x = tf(load_image(image_src)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        vals, idx = probs.topk(topk)
        return [{"scene": cats[i], "confidence": float(v)} for v, i in zip(vals, idx)]


# ---------------------------------------------------------------- timm ViT
def _timm_vit():
    """The timm ViT-B/16 (ImageNet-1k head) shared by classification and
    semantic feature extraction, with its preprocessing transform."""
    def build():
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        return model, timm.data.create_transform(**cfg)
    return _cached_model("timm_vit", build)


class TimmClassifyTool(_Tool):
    name = "classify_image_timm"
    deps = ["timm", "torch"]
    description = "Classify the image content/scene via a timm ViT (top ImageNet labels)."

    def run(self, image_src: Any, topk: int = 3) -> List[Dict[str, Any]]:
        import torch
        from timm.data import ImageNetInfo
        model, tf = _timm_vit()
        x = tf(load_image(image_src)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        vals, idx = probs.topk(topk)
        info = ImageNetInfo()
        return [{"label": info.index_to_description(int(i)).split(",")[0], "confidence": float(v)}
                for v, i in zip(vals, idx)]


class ViTFeatureTool(_Tool):
    name = "extract_features_vit"
    deps = ["timm", "torch"]
    description = "Extract a semantic feature embedding from an image via a timm ViT."

    def run(self, image_src: Any) -> List[float]:
        import torch
        model, tf = _timm_vit()
        x = tf(load_image(image_src)).unsqueeze(0)
        with torch.no_grad():
            feat = model.forward_head(model.forward_features(x), pre_logits=True)[0]
        return [round(float(v), 6) for v in feat.tolist()]   # semantic embedding vector

    @staticmethod
    def observation(vec: List[float]) -> Dict[str, Any]:
        return {"embedding_dim": len(vec)}


# ---------------------------------------------------------------- OpenCV (image properties)
class OpenCVTool(_Tool):
    name = "analyze_image_opencv"
    deps = ["cv2", "numpy"]
    description = ("Decode the image with OpenCV and report basic visual properties "
                   "(width, height, brightness, dominant colors).")

    def run(self, image_src: Any, k: int = 3) -> Dict[str, Any]:
        import cv2
        import numpy as np
        arr = cv2.imdecode(np.frombuffer(image_bytes(image_src), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("OpenCV could not decode the image")
        h, w = arr.shape[:2]
        small = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(labels.flatten(), minlength=k) / len(labels)
        colors = [{"rgb": [int(c) for c in centers[i][::-1]], "share": round(float(share[i]), 3)}
                  for i in np.argsort(-share)]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return {"width": int(w), "height": int(h), "brightness": round(float(gray.mean()) / 255, 3),
                "dominant_colors": colors}


# ---------------------------------------------------------------- spaCy NER (text)
class SpacyNERTool(_Tool):
    name = "extract_entities_spacy"
    deps = ["spacy", "en_core_web_sm"]
    description = "Extract named entities (text, label, offsets) from text via spaCy."

    def _load(self):
        import spacy
        return _cached_model("spacy", lambda: spacy.load("en_core_web_sm"))

    def run(self, text: str) -> List[Dict[str, Any]]:
        nlp = self._load()
        doc = nlp(str(text)[:100000])
        return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]


# ---------------------------------------------------------------- Transformers NER (text)
_HF_LABELS = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}


class TransformersNERTool(_Tool):
    name = "extract_entities_transformers"
    deps = ["transformers", "torch"]
    description = "Extract named entities (text, label, offsets, score) via a Transformers BERT NER model."
    model_name = "dslim/bert-base-NER"

    def _load(self):
        def build():
            from transformers import pipeline
            return pipeline("token-classification", model=self.model_name,
                            aggregation_strategy="simple", device=-1)
        return _cached_model("hf_ner", build)

    def run(self, text: str) -> List[Dict[str, Any]]:
        out = []
        t = str(text)[:5000]
        for e in self._load()(t):
            out.append({"text": t[e["start"]:e["end"]], "label": _HF_LABELS.get(e["entity_group"],
                                                                                e["entity_group"]),
                        "start": int(e["start"]), "end": int(e["end"]),
                        "confidence": round(float(e["score"]), 4)})
        return out


# ---------------------------------------------------------------- GLiNER NER (text)
# GLiNER entity types, reported with spaCy-style labels
_GLINER_TYPES = {"person": "PERSON", "organization": "ORG", "location": "LOC",
                 "work of art": "WORK_OF_ART", "product": "PRODUCT", "event": "EVENT",
                 "date": "DATE"}


class GLiNERNERTool(_Tool):
    """GLiNER (Zaratiana et al., NAACL 2024): generalist NER over open entity
    types, e.g. titles of works that CoNLL-style taggers miss."""
    name = "extract_entities_gliner"
    deps = ["gliner"]
    description = ("Extract named entities, including titles of works and products, with "
                   "scores via GLiNER.")
    model_name = "urchade/gliner_medium-v2.1"
    threshold = 0.5

    def _load(self):
        def build():
            from gliner import GLiNER
            return GLiNER.from_pretrained(self.model_name)
        return _cached_model("gliner", build)

    def run(self, text: str) -> List[Dict[str, Any]]:
        t = str(text)[:5000]
        return [{"text": e["text"], "label": _GLINER_TYPES[e["label"]],
                 "start": int(e["start"]), "end": int(e["end"]),
                 "confidence": round(float(e["score"]), 4)}
                for e in self._load().predict_entities(t, list(_GLINER_TYPES),
                                                       threshold=self.threshold)]


# ---------------------------------------------------------------- VADER (sentiment)
class VaderSentimentTool(_Tool):
    name = "analyze_sentiment_vader"
    deps = ["vaderSentiment"]
    description = "Rule-based sentiment (positive/negative/neutral + compound) via VADER."

    def _load(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return _cached_model("vader", SentimentIntensityAnalyzer)

    def run(self, text: str) -> Dict[str, Any]:
        s = self._load().polarity_scores(str(text)[:5000])
        comp = s["compound"]
        label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
        return {"sentiment": label, "compound": round(comp, 4), "scores": s}


# ---------------------------------------------------------------- KeyBERT (topics)
class KeyBERTTool(_Tool):
    name = "extract_topics_keybert"
    deps = ["keybert"]
    description = "Extract key topics / keyphrases from text via KeyBERT (BERT embeddings)."

    def _load(self):
        from keybert import KeyBERT
        # default sentence-transformers all-MiniLM-L6-v2
        return _cached_model("keybert", KeyBERT)

    def run(self, text: str, top_n: int = 3) -> List[Dict[str, Any]]:
        kws = self._load().extract_keywords(
            str(text)[:5000], keyphrase_ngram_range=(1, 2),
            stop_words="english", top_n=top_n)
        return [{"topic": k, "score": round(float(s), 4)} for k, s in kws]


# spaCy entity label -> our EntityType name
SPACY_LABEL_MAP = {
    "PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "LOCATION", "LOC": "LOCATION",
    "FAC": "LOCATION", "EVENT": "EVENT", "DATE": "TIME", "TIME": "TIME",
    "CARDINAL": "QUANTITY", "QUANTITY": "QUANTITY", "MONEY": "QUANTITY",
    "PRODUCT": "OBJECT", "WORK_OF_ART": "CONCEPT", "NORP": "CONCEPT",
    "MISC": "CONCEPT",            # Transformers CoNLL label
}


# Tools registered by users before query processing (paper Sec. III-A), by name.
_REGISTERED: Dict[str, Any] = {}

# kinds of metadata a registered tool may declare, per modality, with the
# output its ``run`` returns
TOOL_KINDS = {
    "text": ("ner",        # [{"text", "label", "start", "end", "confidence"?}]
             "topic",      # [{"topic", "score"?}]
             "other"),     # any JSON value, kept per document
    "image": ("object_detection",   # [{"label", "confidence", "bbox", "area", "saliency"}]
              "scene",              # [{"scene", "confidence"}]
              "other"),             # any JSON value, kept per image
}
_BUILTIN = ("yolo", "detectron2", "mmdetection", "places365", "timm", "vit", "opencv",
            "spacy", "transformers_ner", "gliner", "vader", "keybert")


def register_tool(modality: str, name: str, tool: Any, kind: str = "other") -> Any:
    """Register a further extraction tool before query processing.

    ``tool`` needs ``run(item)`` (the paragraph text for a text tool, the PIL
    image for an image tool) and a ``description`` for the ReAct agent;
    ``available`` defaults to True.  ``kind`` declares the metadata it returns
    (``TOOL_KINDS``): its output is merged with that of the built-in tools of
    the same kind, e.g. the detections of an ``object_detection`` tool become
    entity mentions.  The processors offer the tool to the LLM under ``name``.
    """
    if modality not in TOOL_KINDS:
        raise ValueError(f"modality must be one of {sorted(TOOL_KINDS)}, got {modality!r}")
    if kind not in TOOL_KINDS[modality]:
        raise ValueError(f"a {modality} tool is one of {TOOL_KINDS[modality]}, got {kind!r}")
    if not name or name in _BUILTIN:
        raise ValueError(f"{name!r} is no free tool name")
    if not callable(getattr(tool, "run", None)):
        raise ValueError("a tool needs a run(item) method")
    tool.name, tool.modality, tool.kind = name, modality, kind
    for attr, default in (("description", f"user-registered {modality} tool ({kind})"),
                          ("available", True), ("_err", None)):
        if not hasattr(tool, attr):
            setattr(tool, attr, default)
    _REGISTERED[name] = tool
    return tool


def unregister_tool(name: str) -> None:
    _REGISTERED.pop(name, None)


def registered_tools(tools: Dict[str, Any], modality: str) -> List[Any]:
    """The available user-registered tools of a modality among ``tools``."""
    return [t for t in tools.values()
            if getattr(t, "modality", None) == modality and getattr(t, "available", False)]


def all_tools() -> Dict[str, "_Tool"]:
    """Every registered tool -- built-in and user-registered -- keyed by the
    name processors require it by."""
    return {"yolo": YOLOTool(), "detectron2": Detectron2Tool(),
            "mmdetection": MMDetectionTool(), "places365": Places365Tool(),
            "timm": TimmClassifyTool(), "vit": ViTFeatureTool(), "opencv": OpenCVTool(),
            "spacy": SpacyNERTool(), "transformers_ner": TransformersNERTool(),
            "gliner": GLiNERNERTool(), "vader": VaderSentimentTool(), "keybert": KeyBERTTool(),
            **_REGISTERED}


def available_tools() -> Dict[str, Any]:
    """Report which external tools can actually load (for honest run manifests)."""
    out: Dict[str, Any] = {}
    for t in all_tools().values():
        ok = t.available
        out[t.name] = ok
        if not ok:
            out[t.name + "_error"] = t._err
    return out
