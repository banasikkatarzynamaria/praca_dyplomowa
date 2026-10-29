
import requests

BASE_URL = "http://127.0.0.1:8080"

def test_api():
    print("=" * 50)
    print("Testing Leak-Free Breast Cancer Pipeline API...")
    print("=" * 50)

    # 1. Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print("1. Root Endpoint Test Status:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print(f"Error connecting to root: {e}")
        return

    print("-" * 50)

    # 2. Test prediction endpoint using raw 30-feature sample from notebook or fallback
    try:
        sample_features = None
        for var_name in ['X_test', 'X']:
            if var_name in globals():
                data_source = globals()[var_name]
                sample_features = data_source.iloc[0].tolist() if hasattr(data_source, "iloc") else data_source[0].tolist()
                print(f"Pulled sample from notebook variable: '{var_name}'")
                break

        if sample_features is None:
            # Fallback 30 raw features sample
            sample_features = [
                14.27, 22.55, 93.74, 620.1, 0.1048, 0.13, 0.1058, 0.05858, 0.1943, 0.06155,
                0.2656, 0.9831, 1.956, 24.08, 0.00544, 0.0245, 0.03488, 0.01053, 0.01721, 0.003198,
                15.29, 28.76, 103.1, 706.0, 0.1383, 0.2882, 0.2814, 0.1389, 0.3159, 0.0762
            ]

        payload = {"features": sample_features}
        response = requests.post(f"{BASE_URL}/predict", json=payload)

        print("2. Prediction Endpoint Test Status:", response.status_code)
        print("Prediction Response:", response.json())
    except Exception as e:
        print(f"Error testing prediction: {e}")

    print("=" * 50)

if __name__ == "__main__":
    test_api()
