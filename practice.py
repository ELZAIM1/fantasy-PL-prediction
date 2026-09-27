import requests 
url = f"https://fantasy.premierleague.com/api/event/{5}/live/"
try:
    # 3. Send a GET request to the endpoint
    response = requests.get(url)
    
    # 4. Check if the request was successful (Status Code 200)
    if response.status_code == 200:
        # 5. Parse the response data as JSON
        data = response.json()
        print("Data retrieved successfully:")
        print(data["elements"][1]["stats"])
        # for i in range(1,667) : 
        #     print(f"{i} - {data["elements"][i]}")
    else:
        print(f"Failed to get data. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"A network error occurred: {e}")