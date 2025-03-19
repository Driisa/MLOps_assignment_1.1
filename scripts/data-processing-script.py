#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for penguin classification.
This script downloads the penguin dataset from seaborn and transforms it into a SQLite database.
"""

import os
import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    Path("data").mkdir(exist_ok=True)
    Path("data/predictions").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

def load_penguin_data():
    """Load penguin dataset from seaborn."""
    print("Loading penguin dataset...")
    # Load the penguins dataset from seaborn
    penguins = sns.load_dataset("penguins").dropna()
    print(f"Loaded {len(penguins)} penguin samples with complete data.")
    return penguins

def create_database(penguins_df):
    """
    Create SQLite database with the penguin data according to the specified schema.
    
    Schema:
    - PENGUINS: species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, island_id, animal_id
    - ISLANDS: island_id, name
    """
    print("Creating SQLite database...")
    
    # Create database connection
    db_path = "data/penguins.db"
    conn = sqlite3.connect(db_path)
    
    try:
        # Create ISLANDS table and populate it
        islands = penguins_df['island'].unique()
        islands_df = pd.DataFrame({
            'island_id': range(1, len(islands) + 1),
            'name': islands
        })
        
        islands_df.to_sql('ISLANDS', conn, if_exists='replace', index=False)
        print(f"Created ISLANDS table with {len(islands_df)} records.")
        
        # Create island_id mapping
        island_mapping = dict(zip(islands_df['name'], islands_df['island_id']))
        
        # Add island_id to penguins dataframe
        penguins_df['island_id'] = penguins_df['island'].map(island_mapping)
        
        # Add animal_id (just a unique identifier for each penguin)
        penguins_df['animal_id'] = range(1, len(penguins_df) + 1)
        
        # Select and rename columns according to the schema
        penguins_for_db = penguins_df[[
            'species', 'bill_length_mm', 'bill_depth_mm', 
            'flipper_length_mm', 'body_mass_g', 'sex', 
            'island_id', 'animal_id'
        ]]
        
        # Save to database
        penguins_for_db.to_sql('PENGUINS', conn, if_exists='replace', index=False)
        print(f"Created PENGUINS table with {len(penguins_for_db)} records.")
        
        # Create database indices for better performance
        cursor = conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON PENGUINS(species)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_island_id ON PENGUINS(island_id)')
        conn.commit()
        
        print("Database created successfully at:", db_path)
        
        # Print sample data to verify
        print("\nSample data from PENGUINS table:")
        print(pd.read_sql("SELECT * FROM PENGUINS LIMIT 5", conn))
        
        print("\nSample data from ISLANDS table:")
        print(pd.read_sql("SELECT * FROM ISLANDS", conn))
        
    finally:
        conn.close()

def main():
    """Main function to prepare data."""
    create_directories()
    penguins_df = load_penguin_data()
    
    # Display basic info about the dataset
    print("\nDataset Information:")
    print(penguins_df.info())
    
    print("\nSummary Statistics:")
    print(penguins_df.describe())
    
    # Count species
    print("\nSpecies Distribution:")
    print(penguins_df['species'].value_counts())
    
    # Create and populate the database
    create_database(penguins_df)
    
    print("\nData preparation completed successfully!")

if __name__ == "__main__":
    main()