import json
import redis

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# Clear old database
r.flushdb()

with open('food_nutrition.json', 'r') as file:
    data = json.load(file)

    # If wrapped with food_classes
    items = data.get("food_classes", data)

    # Store ENTIRE dictionary under ONE key
    r.set("food_details", json.dumps(items))

print("Stored all food nutrition data inside Redis key: food_details")