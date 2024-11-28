import requests

for n in range(1,120):
    image_url = f'https://limitlesstcg.nyc3.digitaloceanspaces.com/one-piece/OP08/OP08-{n:03}_EN.webp'
    img_data = requests.get(image_url).content
    with open(f'Cards/OP08-{n:03}.jpg', 'wb') as handler:
        handler.write(img_data)