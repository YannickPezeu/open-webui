#!/usr/bin/env python3
"""
Script pour tester directement l'endpoint RAG sans passer par l'outil
"""

import requests
import json
import time

def test_endpoint():
    url = "http://localhost:8080/api/v1/tools/perso/rag-search"
    
    # Payload minimal
    payload = {
        "query": "test simple",
        "files": []
    }
    
    headers = {
        "Content-Type": "application/json",
        # Ajoutez votre token d'auth si nécessaire
        # "Authorization": "Bearer YOUR_TOKEN"
    }
    
    print(f"🔍 Test de l'endpoint: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print("⏳ Envoi de la requête...")
        start_time = time.time()
        
        response = requests.post(
            url, 
            json=payload, 
            headers=headers,
            timeout=10  # Timeout court pour tester
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Réponse reçue en {duration:.2f}s")
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ JSON Response: {json.dumps(result, indent=2)}")
            except:
                print(f"⚠️  Response (text): {response.text}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - L'endpoint ne répond pas")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - L'endpoint n'est pas accessible")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_endpoint_health():
    """Test si le serveur répond du tout"""
    base_url = "http://localhost:8080"
    
    # Tester d'abord si le serveur répond
    health_endpoints = [
        "/health",
        "/",
        "/api/v1/",
        "/docs"
    ]
    
    print("🔍 Test de santé du serveur...")
    
    for endpoint in health_endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"   Tentative: {url}")
            response = requests.get(url, timeout=5)
            print(f"   ✅ {endpoint}: {response.status_code}")
            return True
        except:
            print(f"   ❌ {endpoint}: pas de réponse")
    
    print("❌ Le serveur ne semble pas accessible")
    return False

if __name__ == "__main__":
    print("=== TEST ENDPOINT RAG ===\n")
    
    # 1. Test basique du serveur
    if not test_endpoint_health():
        print("\n🚨 Le serveur OpenWebUI ne répond pas sur localhost:8080")
        exit(1)
    
    print("\n" + "="*50 + "\n")
    
    # 2. Test de l'endpoint RAG
    test_endpoint()
    
    print("\n=== FIN DES TESTS ===")