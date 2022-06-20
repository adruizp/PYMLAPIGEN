def test_prueba():
    assert 2==2



def test_request_example(client):
    response = client.get("/")
    assert b"<title>  Home </title>" in response.data


def test_request_example2(client):
    response = client.get("/api/")
    b'{"status":"Api is working","generatedApis":{}}\n' in response.data