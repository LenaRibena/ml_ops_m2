import time
from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def get_root(self) -> None:
        self.client.get("/")
    
    @task(3)
    def get_item(self) -> None:
        for i in range(1, 10):
            self.client.get(f"/items/{i}", name = "/items")
            time.sleep(1)
            