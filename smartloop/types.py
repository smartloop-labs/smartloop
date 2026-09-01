
import uuid
from enum import Enum

class LLMOptions:
	temperature = 0.3

class ModelType(Enum):
    TEXT = "text"
    MULTI_MODAL = "multi-modal"

class ModelCapability(str, Enum):
    GENERAL = "general"
    SKILLS = "skills"
    DP = "dp"
    VISION = "vision"
    CODING = "coding"
    # Declares that the model's chat template reads ``enable_thinking`` — i.e.
    # it can reason in a hidden block before answering.  Unlike the others this
    # is not a routing target (no step asks for a "thinking model"); it is the
    # capability the model picker reads to show the toggle, and what gates the
    # per-request ``thinking`` flag on /v1/chat/completions.  See
    # ``SLPModel.thinking_default`` for whether it starts on.
    THINKING = "thinking"
	
class Document:
	def __init__(self, **kwargs) -> None:
		self.text = kwargs.get('text', "")
		self.confidence = kwargs.get('confidence', 0.0)
		self.id = kwargs.get('id', str(uuid.uuid4()))
	
	def to_dict(self) -> dict:
		return dict(
			text=self.text,
			confidence=self.confidence,
			id=self.id
		)
