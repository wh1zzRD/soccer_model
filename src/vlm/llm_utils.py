import os
#os.environ["HF_TOKEN"] 
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import pandas as pd
import re
import gc


def _finalize_commentary(commentary: str, action_label: str) -> str:
    commentary = re.sub(r"[*_`#\[\]{}]", "", commentary)
    commentary = re.sub(r"\s+", " ", commentary).strip()
    commentary = re.sub(r"^commentary:\s*", "", commentary, flags=re.IGNORECASE)

    if action_label.lower() == "goal":
        commentary = re.sub(r"\bG+O+A+L+\b!*", "GOAL!", commentary, flags=re.IGNORECASE)
        commentary = re.sub(r"\s*\bGOAL!\s*", " ", commentary, flags=re.IGNORECASE)
        words = commentary.strip(" .!-").split()
        commentary = " ".join(words[:18])
        if commentary:
            return f"{commentary}... GOAL!"
        return "GOAL!"

    commentary = re.sub(r"\bG+O+A+L+\b!*", "", commentary, flags=re.IGNORECASE)
    commentary = re.sub(r"\bgoal\b!*", "", commentary, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", commentary).strip(" .!-")


def _split_short_cue(text: str, max_words: int = 7) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if text else []

    first = " ".join(words[:max_words])
    second = " ".join(words[max_words:])
    return [part for part in [first, second] if part]


def _build_timed_commentary(commentary: str, action_label: str) -> list[dict]:
    """Build conservative timing cues for a 10s clip.

    The action clip is cut around the detected event, so the event itself is
    usually close to the middle of the clip. For goals we hard-anchor the goal
    call to that event time and let the LLM only provide the short buildup text.
    """
    if action_label.lower() == "goal":
        goal_time = float(os.environ.get("SOCCER_MODEL_GOAL_CUE_TIME", "5.0"))
        buildup = re.sub(r"\s*\bG+O+A+L+\b!*\s*", " ", commentary, flags=re.IGNORECASE)
        buildup = re.sub(r"\s+", " ", buildup).strip(" .!-")
        parts = _split_short_cue(buildup, max_words=7)

        cues = []
        if len(parts) == 1:
            cues.append({"time": max(0.4, goal_time - 2.0), "text": parts[0]})
        elif len(parts) >= 2:
            cues.append({"time": max(0.4, goal_time - 3.4), "text": parts[0]})
            cues.append({"time": max(0.8, goal_time - 1.7), "text": parts[1]})

        cues.append({"time": goal_time, "text": "GOAL!"})
        return cues

    sentences = [
        sentence.strip(" .!-")
        for sentence in re.split(r"(?<=[.!?])\s+|;\s+|\.\.\.\s*", commentary)
        if sentence.strip(" .!-")
    ]
    if not sentences:
        return []

    times = [0.6, 3.0, 5.5, 7.2]
    return [
        {"time": times[min(index, len(times) - 1)], "text": sentence}
        for index, sentence in enumerate(sentences[:4])
    ]



class Commentator:
    def __init__(self):
        """
        Initialize the text model for commentary generation.
        """
        model_id = os.environ.get("SOCCER_MODEL_COMMENTARY_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading model (16-bit mode activated)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).eval().to("cuda")

    def rewrite_text(self, prompt, max_length=48):
        """
        Reformulates a detailed VLM description into a concise and professional sports commentary.
        """
        print("Generating commentary...")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful sports commentator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                repetition_penalty=1.08,
            )
            generation = generation[0][input_len:]
        
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        print(f"Commentary: {decoded}\n")

        return decoded


def generate_commentary(descriptions, match_context):
    """Converts VLM descriptions into short spoken soccer commentary."""
    results = {}
    commentator = Commentator()
    try:
        for path, data in descriptions.items():
            with torch.no_grad():
                action_label = data["label"]
                action_description = data["description"]
                if action_label.lower() == "goal":
                    length_rule = "Output exactly one commentary, 12 to 20 words. The last words must be GOAL!"
                    timing_rule = "Keep the buildup extremely short so GOAL lands at the scoring moment, not after it."
                    example = "He drives into space, strikes it cleanly... GOAL!"
                else:
                    length_rule = "Output exactly one commentary, 14 to 26 words."
                    timing_rule = "Keep it short enough for a 10-second clip."
                    example = "He moves it wide, pressure arrives, and the ball is cleared."

                print(f"\n🎙 Generating commentary for {action_label}...")

                llm_prompt = f"""
                You are writing a short live soccer commentary line for text-to-speech.

                Detected event label: {action_label}

                The visual description below may contain repeated or uncertain observations.
                Use it only as a rough guide. Be conservative and do not add details unless
                they are strongly supported.

                Style:
                - Sound like an excited live TV soccer commentator.
                - Build urgency with short phrases and one emotional peak.
                - For a Goal, celebrate strongly with the word GOAL.

                Rules:
                - {length_rule}
                - No timestamps, no labels, no markdown.
                - Do not mention post, crossbar, rebound, repeated saves, player names,
                  team names, or crowd unless they are clearly and consistently visible.
                - Do not describe contradictory actions.
                - Preserve timing: describe buildup first, then the decisive action.
                - If the detected event is Goal, the goal call must be the final sentence,
                  not the first sentence. Say GOAL! once at the end.
                - If the detected event is not Goal, do not say goal.
                - {timing_rule}

                Example style:
                "{example}"

                Raw visual description:
                {action_description}
                """

                commentary = _finalize_commentary(commentator.rewrite_text(llm_prompt), action_label)
                timed_commentary = _build_timed_commentary(commentary, action_label)
                print(f"len de commentary = {len(commentary.split(' '))}")
                results[path] = {
                    "Action": action_label,
                    "Raw Description": action_description,
                    "Generated Commentary": commentary,
                    "Timed Commentary": timed_commentary,
                }
                del commentary, timed_commentary, llm_prompt
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        del commentator
        torch.cuda.empty_cache()
        gc.collect()

    return results

# Example Usage:
if __name__ == "__main__":
    # Example match context
    match_context = "It's a Premier League match between Tottenham and Chelsea. Tottenham plays with white jerseys, and Chelsea plays with dark blue jerseys."

    # Generate commentary and save to CSV
    #results = generate_commentary(DESCRIPTIONS_OUTPUT, match_context)

    # Print the generated commentaries
    #print("\n📋 Generated Commentaries:")
    #print(results)
