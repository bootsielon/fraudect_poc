"""
This script checks the structure of the fraud pipeline to identify issues.
"""

import importlib
import inspect
import sys
from pprint import pprint

def check_pipeline_structure():
    """Check the structure of the pipeline and identify issues."""
    
    try:
        # Import the pipeline module
        from ml_pipeline.base import BasePipeline
        pipeline_module = importlib.import_module('ml_pipeline.pipeline')
        
        # Find all pipeline classes
        pipeline_classes = [cls for _, cls in inspect.getmembers(pipeline_module, inspect.isclass)
                           if issubclass(cls, BasePipeline) and cls != BasePipeline]
        
        print(f"Found {len(pipeline_classes)} pipeline classes: {[cls.__name__ for cls in pipeline_classes]}")
        
        # Check each pipeline class
        for cls in pipeline_classes:
            print(f"\nChecking {cls.__name__}:")
            
            # Create an instance
            instance = cls()
            
            # Check if run_all method is implemented
            if hasattr(instance, 'run_all'):
                print("  ✓ Has run_all method")
                # Get source code
                try:
                    run_all_source = inspect.getsource(instance.run_all)
                    print(f"  Run all method source:\n{run_all_source}")
                except Exception as e:
                    print(f"  ! Error getting source: {e}")
            else:
                print("  ✗ Missing run_all method")
            
            # Check if feature engineering is included in the pipeline
            methods = [name for name, _ in inspect.getmembers(instance, inspect.ismethod)]
            fe_method = next((m for m in methods if 'feature' in m.lower()), None)
            
            if fe_method:
                print(f"  ✓ Has feature engineering method: {fe_method}")
            else:
                print("  ✗ No obvious feature engineering method found")
                
            # Check step methods
            step_methods = [m for m in methods if 'run_step' in m or 'step' in m]
            print(f"  Step methods: {step_methods}")
            
            # Try to run methods to check dataframes
            print("  Checking dataframes after steps:")
            for method_name in ['run_eda', 'run_step1']:
                if hasattr(instance, method_name):
                    try:
                        print(f"    Running {method_name}...")
                        getattr(instance, method_name)()
                        print(f"    After {method_name}, dataframes: {list(instance.dataframes.keys())}")
                    except Exception as e:
                        print(f"    ! Error running {method_name}: {e}")
    
    except Exception as e:
        print(f"Error checking pipeline structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pipeline_structure()
