from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from data_generator import generate_transaction_data

def train_fraud_detector():
    """
    Train a simple Random Forest model to detect suspicious transactions
    """
    # Generate data
    data = generate_transaction_data()
    
    # Prepare features and target
    X = data.drop('is_suspicious', axis=1)
    y = data['is_suspicious']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Model saved as fraud_model.pkl")
    
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_fraud_detector()