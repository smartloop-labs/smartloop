
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
