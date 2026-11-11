
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """
    Represents an individual factual claim extracted
    from an AI-generated response.
    """

    claim_id: str = Field(
        ...,
        description="Unique identifier for the claim."
    )

    text: str = Field(
        ...,
        min_length=1,
        description="The factual statement that needs verification."
    )
