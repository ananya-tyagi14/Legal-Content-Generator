import time
import pandas as pd
from requests.exceptions import HTTPError, ConnectTimeout
from PySide6.QtCore import QThread, Signal


class TrendsWorker(QThread):

    """
    QThread subclass to fetch Google Trends data asynchronously.
    Emits:
      - finished(region_dataframe, combined_queries_dataframe)
      - error(error_message)
    """

    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, keywords, pytrends, trends, parent=None):

        """
        Initialize the worker thread.
        Args:
            keywords (list of str): List of keywords to query.
            pytrends: Initialized pytrends request object for interest_by_region.
            trends: Initialized pytrends request object for related_queries.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.keywords = keywords
        self.pytrends = pytrends
        self.trends = trends
  

    def run(self):

        """
        Entry point for the thread. 
        
        """
        try:
            # Introduce a short sleep to avoid immediate requests
            time.sleep(5.0)
            # Build payload for interest_by_region for all keywords,
            self.pytrends.build_payload(self.keywords, timeframe="today 1-m", geo='GB')
            regiondf = self.pytrends.interest_by_region()

        except Exception as e:
            self.error.emit(f"Could not fetch region data:\n{e}")
            return

        # Filter out regions where all keyword    
        regiondf = regiondf[(regiondf != 0).all(axis=1)]
        # Drop any rows that are entirely NaN (to clean up the DataFrame)
        regiondf.dropna(how='all', axis=0, inplace=True)
        #Sort regions descending by interest for the first keyword and keep top 10
        top10 = regiondf.sort_values(by=self.keywords[0], ascending=False).head(10)

        frames = []

        #Iterate over each keyword to fetch related queries
        for kw in self.keywords:
            time.sleep(5.0)
            try:
                # Fetch related queries for the current keyword
                result = self.trends.related_queries(kw)

            # If there is a network or HTTP error, skip this keyword
            except (HTTPError, ConnectTimeout):
                continue
                
            except Exception:
                continue

            # Extract 'top' and 'rising' DataFrames from the result
            top_df = result.get('top')
            rising_df = result.get('rising')
                
            if (top_df is None or top_df.empty) and (rising_df is None or rising_df.empty):
                    continue

            # Convert lists/dicts to pandas DataFrames if necessary
            top_df = pd.DataFrame(top_df) if top_df is not None else pd.DataFrame()
            rising_df = pd.DataFrame(rising_df) if rising_df is not None else pd.DataFrame()

            # Reset indices to ensure alignment
            top_df = top_df.reset_index(drop=True)
            rising_df = rising_df.reset_index(drop=True)
            
            max_rows = max(len(top_df), len(rising_df))   # Determine the maximum number of rows between top and rising
            top_df = top_df.reindex(range(max_rows))
            rising_df = rising_df.reindex(range(max_rows))

            # Rename columns for clarity:
            top_df    = top_df.rename(columns={
                'query': 'top query',
                'value': 'top query value'
            })
            rising_df = rising_df.rename(columns={
                'query': 'related query',
                'value': 'related query value'
            })

            # Concatenate the two DataFrames horizontally
            df_all = pd.concat([top_df, rising_df], axis=1)
             # Insert a 'keyword' column at position 0 to identify which keyword these rows belong to
            df_all.insert(0, 'keyword', kw)
            frames.append(df_all)

        # 6. Concatenate all keyword-specific DataFrames into one big DataFrame, ignoring index
        allq = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        #Emit the finished signal with the top10 regions and the combined queries DataFrame
        self.finished.emit(top10, allq)

                

