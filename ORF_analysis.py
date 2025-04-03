import os
import warnings
import argparse
import pandas as pd
from CalculateWeight import NeuroAnalyzer
from NeuroDataConfig import NeuroConfig, NeuroDataLoader
from NeuroPlotter import ORFPlotter
from tqdm import tqdm
from datetime import datetime
import gc
from multiprocessing import Pool, cpu_count

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_neuron_types(config: NeuroConfig) -> list:
    """Get all neuron types from visual_neuron_types.txt"""
    visual_neurons_path = os.path.join(config.DATA_DIR, config.VISUAL_NEURONS_FILE)
    df = pd.read_csv(visual_neurons_path)
    neuron_types = df['type'].unique().tolist()
    print(f"Total {len(neuron_types)} neuron types found, starting analysis...")
    return neuron_types, df  # Return DataFrame for later use

def process_neuron_type(source_type: str, target_type: str, config: NeuroConfig, visual_neurons_df: pd.DataFrame, connection_map: dict):
    """Task function to process single source neuron type to target neuron type"""
    try:
        # Update configuration
        config.SOURCE_NEURON_TYPE = source_type
        config.TARGET_NEURON_TYPE = target_type
        config.TARGET_COLUMN = "cell_type"
        config.RESULT_ROOT = f"./result/visual_neuron_ORF/{source_type}"
        
        # Get target neuron IDs with right side for current target type
        target_ids = visual_neurons_df[
            (visual_neurons_df['type'] == target_type) & 
            (visual_neurons_df['side'] == 'right')
        ]['root_id'].tolist()
        
        # Analyze connections using the same connection map
        analyzer = NeuroAnalyzer(config)
        analyzer.data_loader.target_ids = target_ids
        is_exist = analyzer.analyze_connections(connection_map)
        if is_exist:
            print(f"✅ Connection exists from {source_type} to {target_type}")
            del analyzer  # Remove unused object
            gc.collect()  # Force garbage collection
            return
        
        # Perform visualization
        plotter = ORFPlotter(config)
        plotter.data_loader.target_ids = target_ids
        plotter.plot_receptive_field()
        
        print(f"✅ Processing completed from {source_type} to {target_type}")
    except Exception as e:
        print(f"❌ Error processing from {source_type} to {target_type}: {str(e)}")
    finally:
        gc.collect()  # Force garbage collection

def main(source_neuron_types: list):
    """Main function: perform neuron analysis and visualization"""
    # Initialize configuration
    config = NeuroConfig()
    
    print(f"Current time {current_time}")
    
    # Get all neuron types
    neuron_types, visual_neurons_df = get_neuron_types(config)
    
    # Build connection map
    analyzer = NeuroAnalyzer(config)
    connection_map = analyzer.build_connection_map()
    
    if connection_map is None:
        print("Error: Failed to build connection map")
        return
    
    # Set number of parallel processes (adjusted based on CPU cores)
    num_processes = min(cpu_count(), 16)
    print(f"Using {num_processes} processes for parallel processing...")
    
    # Use process pool for parallel processing
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            process_neuron_type,
            [(source_type, target_type, config, visual_neurons_df, connection_map)
             for source_type in source_neuron_types
             for target_type in neuron_types]
        )
    
    print("\n✅ All neuron types processed successfully!")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Neuron connection analysis tool")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source neuron type(s), supports multiple types (comma-separated), e.g.: L1,L2,L3"
    )
    args = parser.parse_args()
    
    # Parse source neuron types
    source_neuron_types = args.source.split(",")
    print(f"Input source neuron types: {source_neuron_types}")
    
    # Call main function
    main(source_neuron_types)
