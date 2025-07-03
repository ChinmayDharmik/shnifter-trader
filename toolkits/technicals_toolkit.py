from core.data_models import ShnifterData
import pandas as pd

class TechnicalAnalysisToolkit:
    """
    A toolkit for performing technical analysis on standardized ShnifterData.
    """
    @staticmethod
    def calculate_sma(data: ShnifterData, length: int) -> ShnifterData:
        """Calculates the Simple Moving Average on the 'close' price."""
        if 'close' not in data.results.columns:
            data.warnings.append("SMA calculation failed: 'close' column not found.")
            return data
        df = data.results.copy()
        df[f'SMA_{length}'] = df['close'].rolling(window=length).mean()
        return ShnifterData(results=df, provider=data.provider)
