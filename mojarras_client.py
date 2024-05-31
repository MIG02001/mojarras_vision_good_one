from requests import Response, post

class MojarrasClient:
    
    def __init__(self, R1, R2, R3, R4) -> None:
        self.uri = "http://10.25.99.133:8080/"
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
    
    def send_cars_request(self) -> Response:
        print(self.uri)
        print(self.R1, self.R2, self.R3, self.R4)
        response = post(self.uri + "algorithm/traffic", json={"cars": [self.R1, self.R2, self.R3, self.R4]}).json()
        print(response)
        return response