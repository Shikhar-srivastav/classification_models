import os
import pandas as pd
from datetime import datetime

output_dir = "testing"
df = pd.read_csv("test_data.csv")

if len(df) > 1000:
    sample_df = df.sample(n=1000)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

    sample_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")