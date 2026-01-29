# test_kaggle_loader.py

import kagglehub
from kagglehub import KaggleDatasetAdapter
from src.datasets.datasets import load_kaggle_othello_games, parse_othello_game

def test_kaggle_dataset():
    """Test loading and parsing Kaggle Othello dataset."""
    
    print("=" * 60)
    print("Testing Kaggle Othello Dataset Loader")
    print("=" * 60)
    
    # First, let's see what's in the dataset
    print("\n1. Loading raw dataset...")
    # Download the dataset (cached locally after first download)
    dataset_path = kagglehub.dataset_download("andrefpoliveira/othello-games")
    import os
    import pandas as pd
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in dataset. Files: {os.listdir(dataset_path)}")
    csv_file = files[0]  # Use the first CSV file found
    print(f"   Found CSV file: {csv_file}")
    
    # Load the CSV file directly using pandas since we already have the path
    csv_path = os.path.join(dataset_path, csv_file)
    df = pd.read_csv(csv_path)
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First few rows:")
    print(df.head())
    
    # Test parsing a single game
    print("\n2. Testing game parsing...")
    if len(df) > 0:
        first_game = df.iloc[0]
        print(f"   First game columns: {dict(first_game)}")
        
        # Try to find moves column
        moves_col = None
        for col in ['game_moves', 'moves', 'game', 'transcript', 'notation']:
            if col in first_game:
                moves_col = col
                break
        
        if moves_col:
            moves_str = str(first_game[moves_col])
            print(f"   Moves ({moves_col}): {moves_str[:100]}...")
            
            try:
                positions = parse_othello_game(moves_str, board_size=8)
                print(f"   ✓ Parsed {len(positions)} positions")
                
                if positions:
                    print(f"   First position grid:\n{positions[0]['grid']}")
            except Exception as e:
                print(f"   ✗ Failed to parse: {e}")
        else:
            print(f"   ✗ No moves column found!")
    
    # Test full loader
    print("\n3. Testing full loader...")
    cfg = {
        'board_size': 8,
        'n_positions': 100,
        'min_move_number': 4,
        'max_move_number': 50,
        'seed': 42,
    }
    
    try:
        positions = load_kaggle_othello_games(cfg)
        print(f"   ✓ Loaded {len(positions)} positions")
        
        if positions:
            print(f"   Sample position:")
            print(f"     Grid shape: {positions[0]['grid'].shape}")
            print(f"     Player: {positions[0]['player']}")
            print(f"     Move number: {positions[0].get('move_number', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_kaggle_dataset()