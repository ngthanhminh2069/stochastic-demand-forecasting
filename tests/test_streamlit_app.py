from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

sys.stdout.reconfigure(encoding="utf-8")

def test_all_steps_and_languages():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    print(f"Testing Streamlit app across all steps and languages using {app_path}...")
    
    # 1. Test Vietnamese across all steps (0 to 8)
    for step in range(9):
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        if len(at.exception) > 0:
            print(f"FAILED on initial load for step {step}:", at.exception)
            sys.exit(1)
        
        at.radio[1].set_value(step)
        at.run()
        if len(at.exception) > 0:
            print(f"FAILED on Step {step} (Tiếng Việt):", at.exception[0].message)
            print(at.exception[0].stack_trace)
            sys.exit(1)
        print(f"  Tiếng Việt - Step {step} passed!")

    # 2. Test English across all steps (0 to 8)
    print("\nTesting English mode across all steps...")
    for step in range(9):
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        at.radio[0].set_value("English")
        at.run()
        if len(at.exception) > 0:
            print(f"FAILED on English switch for step {step}:", at.exception)
            sys.exit(1)

        at.radio[1].set_value(step)
        at.run()
        if len(at.exception) > 0:
            print(f"FAILED on Step {step} (English):", at.exception[0].message)
            print(at.exception[0].stack_trace)
            sys.exit(1)
        print(f"  English - Step {step} passed!")

    # 3. Test changing SKU and Location selectbox
    print("\nTesting SKU and Store selectboxes...")
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    sku_select = at.selectbox[0]
    store_select = at.selectbox[1]
    sku_select.set_value(sku_select.options[1])
    store_select.set_value(store_select.options[1])
    at.run()
    if len(at.exception) > 0:
        print("FAILED on SKU/Store change:", at.exception)
        sys.exit(1)
    print("SKU/Store change: SUCCESS")

    print("\nALL TESTS PASSED! ZERO EXCEPTIONS ACROSS ALL 9 STEPS & 2 LANGUAGES!")

if __name__ == "__main__":
    test_all_steps_and_languages()
