#covert the image into base64 string &
import base64
import os
from pathlib import Path

#this provide the path hint 
from typing import List, Optional
#request send the HTTP reuest to the AI providers
import requests

# Provider chain (free, public, no API key required)
"""
i have configured three AI providers.
Provider	Model
VLM Run	Qwen 3.5 0.8B
OVHcloud	Qwen2.5-VL-72B
BlockRun	Nemotron 3 Nano Omni

The idea is called a Fallback Provider Strategy.
try model 1 if fail move it to next
"""
PROVIDERS = [

    #model1
    {
        "name": "VLM Run Gateway",
        "url": "https://gateway.vlm.run/v1/openai/chat/completions",
        "model": "qwen/qwen3.5-0.8b",
        "key": None,
    },
    #model 2
    {
        "name": "OVHcloud (Qwen2.5-VL)",
        "url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions",
        "model": "Qwen2.5-VL-72B-Instruct",
        "key": None,
    },
    #model 3
    {
        "name": "BlockRun (Nemotron)",
        "url": "https://blockrun.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "key": "not-needed-for-free-models",
    },
]



# this is the 
PROMPT = """
Analyze this image carefully.

Provide the following information:

1. Main objects and subjects.
2. People and their visible activities.
3. Background and environment.
4. Dominant colours and lighting.
5. Visible text or signs.
6. Important details and observations.
7. Overall image summary.

Avoid guessing details that cannot be visually confirmed.
"""



# Vision helpers

def _describe_image_with_provider(
    image_path: str, 
    provider: dict, 
    timeout: int = 90

) -> Optional[str]:
    """Send one image to a single provider. 
    Return text or None on failure."""
   #hare i open the file ib rb(read binray way)
    with open(image_path, "rb") as f:
        # base64 use to encode and decode the file
        #utf8 formate 
        #Utf8 used to  make the understable to cmoputer

        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # https hader 
        # HTTP Headers are additional ingomation sebt '
        # along with an HTTp request they tell the server how 
        # imterpret the request and may provide the authentication
    headers = {"Content-Type": "application/json"} # this mean im sending the jeson formate data
    if provider["key"]:

        """What is Bearer?
        Bearer is an authentication scheme. 
        It tells the API server that you're presenting a
        token to access its services.
        Never expose your real API key in GitHub or frontend code."""

        headers["Authorization"] = f"Bearer {provider['key']}"
    # play load mean actual data sending to the APIA
    payload = {

        #i have used three free models if on faile other work
        "model": provider["model"],
    
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }


# try  to send and fatch the data from the server
    try:
        #send the data via request libairy using post
        resp = requests.post(
            provider["url"],
              headers=headers, 
              json=payload, 
              timeout=timeout
        )

        resp.raise_for_status()
        
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠️  {provider['name']} failed: {e}")
        return None


def _describe_image_online(image_path: str) -> str:
    """Try every provider in order; 
    return the first successful description."""
    for provider in PROVIDERS:
        print(f"  → Trying {provider['name']} …")
        desc = _describe_image_with_provider(image_path, provider)
        
        if desc:
            print(f"  ✅ Success via {provider['name']}")
            return desc
    return "[All vision providers failed — could not analyse this image.]"


# ----------------------------------------------------------------------
# Text file helpers  (replaces the old PDF helpers)
# ----------------------------------------------------------------------
def _build_report_text(user_name: str, summaries: List[tuple[str, str]]) -> str:
    """
    Build the full report as one big string. 
    This is what gets written to the .txt file.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"   Image Summary for {user_name}")
    lines.append("=" * 70)
    lines.append(
        f"Report for: {user_name}   |   Images analysed: {len(summaries)}"
    )
    lines.append("=" * 70)
    lines.append("")

    # One section per image
    for idx, (img_path, desc) in enumerate(summaries, start=1):
        lines.append("-" * 70)
        lines.append(f"Image {idx}: {Path(img_path).name}")
        lines.append(f"Full path : {img_path}")
        lines.append("-" * 70)
        lines.append(desc)
        lines.append("")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main agent
# ----------------------------------------------------------------------
def image_summarize_agent(
    image_paths: List[str],
    user_name: str,
    output_txt: str = "image_summary.txt",
) -> str:
    """
    Summarise each photo with a free online vision model and write a
    personalised TEXT report.
    """
    if not image_paths:
        raise ValueError("At least one image path is required.")

    valid_images = [p for p in image_paths if os.path.isfile(p)]
    if not valid_images:
        raise ValueError("None of the provided image paths exist.")

    # --- 1. Analyse every image -------------------------------------------
    summaries: List[tuple[str, str]] = []
    for img_path in valid_images:
        print(f"\nAnalysing {img_path} …")
        summaries.append((img_path, _describe_image_online(img_path)))

    # --- 2. Build the text report -----------------------------------------
    report_text = _build_report_text(user_name, summaries)

    # --- 3. Write it to disk ---------------------------------------------
    # encoding="utf-8" so em-dashes, curly quotes, etc. survive intact
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✅ Text report written to {output_txt}")
    return output_txt


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    image_summarize_agent(
        image_paths=["/home/vishal/NeuralNest/UI/RAG/agent/img1.png"],
        user_name="Alex",
        output_txt="Alex_photo_summary.txt",
    )