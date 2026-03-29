# -*- coding: utf-8 -*-
"""
Cursor Cloud Agents API 调用工具
"""

import requests
import time
import json

API_KEY = "crsr_ebfefd9f128472e33e5e838547ebe6ebd7a09f3d5760a7713e04bbb72e980fb3"
BASE_URL = "https://api.cursor.com/v0"


def headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {API_KEY}:"
    }


def me():
    r = requests.get(f"{BASE_URL}/me", auth=(API_KEY, ""))
    return r.json()


def list_agents(limit=20):
    r = requests.get(f"{BASE_URL}/agents?limit={limit}", auth=(API_KEY, ""))
    return r.json()


def launch_agent(prompt_text, repo_url, model="claude-4-sonnet-thinking", branch_name=None, auto_create_pr=False):
    data = {
        "prompt": {"text": prompt_text},
        "source": {"repository": repo_url},
        "model": model,
    }
    if branch_name:
        data["target"] = {"branchName": branch_name, "autoCreatePr": auto_create_pr}

    r = requests.post(f"{BASE_URL}/agents", json=data, auth=(API_KEY, ""))
    r.raise_for_status()
    return r.json()


def get_agent(agent_id):
    r = requests.get(f"{BASE_URL}/agents/{agent_id}", auth=(API_KEY, ""))
    return r.json()


def get_conversation(agent_id):
    r = requests.get(f"{BASE_URL}/agents/{agent_id}/conversation", auth=(API_KEY, ""))
    return r.json()


def add_followup(agent_id, prompt_text):
    r = requests.post(
        f"{BASE_URL}/agents/{agent_id}/followup",
        json={"prompt": {"text": prompt_text}},
        auth=(API_KEY, "")
    )
    return r.json()


def stop_agent(agent_id):
    r = requests.post(f"{BASE_URL}/agents/{agent_id}/stop", auth=(API_KEY, ""))
    return r.json()


def wait_for_complete(agent_id, poll_interval=10, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        status = get_agent(agent_id)
        print(f"  Status: {status.get('status')} - {status.get('summary', '...')}")
        if status.get("status") in ("FINISHED", "FAILED", "STOPPED"):
            return status
        time.sleep(poll_interval)
    raise TimeoutError(f"Agent {agent_id} timeout")


def list_models():
    r = requests.get(f"{BASE_URL}/models", auth=(API_KEY, ""))
    return r.json()


def verify_key():
    try:
        info = me()
        print(f"[OK] API Key valid: {info.get('userEmail')}")
        return True
    except Exception as e:
        print(f"[FAIL] API Key invalid: {e}")
        return False


def quick_launch(prompt, repo, model="claude-4-sonnet-thinking", wait=True):
    print(f"[LAUNCH] Starting agent...")
    print(f"  Prompt: {prompt[:100]}")
    print(f"  Repo:  {repo}")
    print(f"  Model: {model}")

    agent = launch_agent(prompt, repo, model=model)
    agent_id = agent["id"]
    print(f"  Agent ID: {agent_id}")
    print(f"  Status: {agent['status']}")

    if wait:
        print(f"\n[WAITING] Waiting for completion...")
        result = wait_for_complete(agent_id)
        print(f"\n[DONE] summary: {result.get('summary', '')}")
        return result
    return agent


if __name__ == "__main__":
    print("=" * 50)
    print("Cursor API Verification")
    print("=" * 50)
    verify_key()

    print("\nAvailable models:")
    try:
        models = list_models()
        print(models)
    except Exception as e:
        print(f"Failed to get models: {e}")

    print("\nRecent agents:")
    try:
        agents = list_agents(5)
        for a in agents.get("agents", []):
            print(f"  {a['id']} - {a['name']} [{a['status']}]")
    except Exception as e:
        print(f"Failed to list agents: {e}")
