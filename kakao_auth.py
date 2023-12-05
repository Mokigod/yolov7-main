import requests
import json

url = 'https://kauth.kakao.com/oauth/token'
client_id = '4b6c91a6c7a4943cb44ba55f081dcfea' # REST API KEY
redirect_uri = 'https://example.com/oauth' # 카카오 디벨로퍼에서 필수 설정항목.
code = '4iu2QN9i5Y-TTyjavx2Sosw4zDuKgrvJY4K22PPAmJxbu5ldY_KlXrL9Qc4KPXOaAAABjDj2BXrUNEQ5evY1pg' 
# https://kauth.kakao.com/oauth/authorize?client_id={client_id}&redirect_uri=https://example.com/oauth&response_type=code&scope=talk_message

data = {
    'grant_type':'authorization_code',
    'client_id':client_id,
    'redirect_uri':redirect_uri,
    'code': code,
    }

response = requests.post(url, data=data)
tokens = response.json()

#발행된 토큰 저장
with open("token.json","w") as kakao:
    json.dump(tokens, kakao)