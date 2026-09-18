
try:
    import sys
    sys.path.append(r"d:\py_project\MRI\code\meta")
    import train_classifier
    print("Syntax check passed")
except ImportError as e:
    print(f"ImportError: {e}")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
except Exception as e:
    print(f"Error: {e}")
