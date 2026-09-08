
import uuid
from enum import Enum

class LLMOptions:
	temperature = 0.3

class ModelType(Enum):
    TEXT = "text"
    MULTI_MODAL = "multi-modal"

class ModelCapability(str, Enum):
    GENERAL = "general"
    # The model that reasons over a set of options before acting: which skill
    # matches this request, which MCP server and tool a step needs, how a
    # prompt decomposes into a plan.  Named for the work rather than for one
    # of its callers — it was ``skills`` when skill matching was the only
    # thing routed to it, which then read as a lie everywhere else, most of
    # all in the SSE that tells a user their MCP tool call picked "skills".
    REASONING = "reasoning"
    DP = "dp"
    VISION = "vision"
    CODING = "coding"
    # Declares that the model's chat template reads ``enable_thinking`` — i.e.
    # it can reason in a hidden block before answering.  Not to be confused
    # with REASONING above, which is a routing target: this one describes how
    # a model answers, that one describes what it is picked for.  Unlike the
    # others this is not a routing target at all; it is the
    # capability the model picker reads to show the toggle, and what gates the
    # per-request ``thinking`` flag on /v1/chat/completions.  See
    # ``SLPModel.thinking_default`` for whether it starts on.
    THINKING = "thinking"

    @classmethod
    def _missing_(cls, value):
        """Resolve names this capability used to go by.

        Capability strings outlive the code that named them: they sit in
        ``config.json``, in a desktop client built against an earlier release,
        and in the model-host protocol between a child and a host that may not
        have been restarted together.  Mapping the old spelling here is what
        keeps any of those from raising instead of routing.
        """
        if isinstance(value, str) and value.strip().lower() == "skills":
            return cls.REASONING
        return None
	
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
