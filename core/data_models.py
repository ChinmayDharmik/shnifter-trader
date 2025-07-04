from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

class ShnifterData(BaseModel):
    """
    Our version of the OBBject. A standardized Pydantic model to hold results
    from any provider, ensuring consistent data structure.
    """
    results: pd.DataFrame = Field(default_factory=pd.DataFrame)
    provider: Optional[str] = Field(default=None, description="The name of the data provider used.")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated during the fetch.")
    
    class Config:
        arbitrary_types_allowed = True # Allow pandas DataFrame

    def to_df(self) -> pd.DataFrame:
        """Convenience function to return the results as a DataFrame."""
        return self.results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convenience function to return results as a dictionary."""
        return self.results.to_dict(orient='records')

class ShnifterNewsData(BaseModel):
    """
    Standardized Pydantic model for news articles from any provider.
    Ensures consistent data structure for sentiment analysis.
    """
    date: datetime = Field(description="The published date of the article.")
    title: str = Field(description="The title of the news article.")
    text: Optional[str] = Field(default=None, description="The body or summary of the news article.")
    url: str = Field(description="The URL to the original article.")
    provider: str = Field(description="The name of the news provider.")
    symbols: Optional[List[str]] = Field(default_factory=list, description="Ticker symbols associated with the article.")

    class Config:
        arbitrary_types_allowed = True
