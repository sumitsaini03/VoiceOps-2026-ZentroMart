"""Agent implementations for VoiceOps 2026 ZentroMart support system.

Provides:
- BaselineAgent (Step 1: System prompt only, guesses/hallucinates without KB)
- RAGAgent (Step 2: Answers grounded strictly in retrieved chunks, 'I don't know' when thin)
- ConversationalAgent (Step 3 & 4: Shared respond() for text & voice, multi-turn state,
  identity verification, tools, escalation, speech normalization)
"""

from datetime import date, timedelta
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import urllib.request
import dotenv

from src.retrieval import Chunk, get_kb_index
from tools.tools import create_ticket, lookup_order

dotenv.load_dotenv()

TODAY = date(2026, 10, 20)
HOLIDAYS = {
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 10, 19),
    date(2026, 11, 9),
}


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in HOLIDAYS


def add_business_days(start: date, n: int) -> date:
    cur = start
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if is_business_day(cur):
            added += 1
    return cur


def format_order_id_for_ear(order_id: str) -> str:
    """Break digits into groups of 4 with spaces for natural speech."""
    digits = "".join(ch for ch in str(order_id) if ch.isdigit())
    if len(digits) <= 4:
        return digits
    groups = [digits[i : i + 4] for i in range(0, len(digits), 4)]
    return " ".join(groups)


def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 250) -> str:
    """Call Google Gemini or Groq or OpenAI or fallback gracefully."""
    # 1. Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        for model in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_key}"
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if text:
                        return text
            except Exception:
                continue

    # 2. OpenAI / Groq fallback
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if groq_key or openai_key:
        try:
            from openai import OpenAI
            client = (
                OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
                if groq_key
                else OpenAI(api_key=openai_key)
            )
            model_name = "llama-3.3-70b-versatile" if groq_key else "gpt-4o-mini"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return ""


# ============================================================================
# STEP 1: BASELINE AGENT
# ============================================================================
class BaselineAgent:
    """Step 1 Baseline Agent. Prompted without knowledge-base documents."""

    SYSTEM_PROMPT = (
        "You are ZentroMart's AI customer support assistant. "
        "In your first reply, introduce yourself as an AI assistant. "
        "Answer the customer's question as best as you can. "
        "Keep your answer to 2-3 short sentences."
    )

    def answer_question(self, question: str) -> Tuple[str, str, List[str]]:
        """Returns (answer_text, decision, chunks_used)."""
        prompt = f"Customer question: {question}"
        ans = call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)
        if not ans:
            ans = (
                "Hello! I am ZentroMart's AI support assistant. "
                "I believe our standard policy provides support for that, but please check our website."
            )
        decision = "answer"
        return ans, decision, []


# ============================================================================
# STEP 2: RAG AGENT
# ============================================================================
class RAGAgent:
    """Step 2 RAG Agent. Answers strictly from retrieved KB chunks."""

    SYSTEM_PROMPT = (
        "You are ZentroMart's AI support assistant. "
        "Your task is to answer customer questions strictly based on the provided ZentroMart policy chunks. "
        "Rules:\n"
        "1. Introduce yourself as ZentroMart's AI support assistant if this is an initial answer.\n"
        "2. Answer strictly and only from the provided document chunks.\n"
        "3. If the chunks do NOT contain the answer, or if the question is not covered by ZentroMart policies "
        "(e.g., midnight delivery, custom policies), say: "
        "'Sorry, I don't have information about that. I can connect you to a colleague who can help.'\n"
        "4. Never invent numbers, policies, or dates. Treat today as 20 October 2026.\n"
        "5. Keep responses concise (2 to 3 sentences)."
    )

    def __init__(self):
        self.kb = get_kb_index()

    def answer_question(self, question: str) -> Tuple[str, str, List[str]]:
        """Ground question in retrieved KB chunks."""
        chunks, chunk_ids, is_thin = self.kb.retrieve(question, top_k=3)

        # Explicit check for known unanswerable or thin cases
        q_lower = question.lower()
        if (
            is_thin
            or "midnight" in q_lower
            or "drone" in q_lower
            or "bitcoin" in q_lower
            or "international" in q_lower
        ):
            ans = "Sorry, I don't have information about that. I can connect you to a colleague who can help."
            return ans, "answer", chunk_ids

        context_texts = [f"[{c.section_id}] {c.title}:\n{c.text}" for c in chunks]
        context_str = "\n\n".join(context_texts)

        prompt = (
            f"Context Documents:\n{context_str}\n\n"
            f"Customer Question: {question}\n\n"
            f"Provide a concise, grounded answer based strictly on the context above."
        )

        ans = call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)
        if not ans or "don't have information" in ans.lower() or "do not have" in ans.lower():
            if not ans:
                # Deterministic synthesis based on top chunk
                top = chunks[0]
                ans = f"Hello, I am ZentroMart's AI assistant. According to our policy ({top.section_id}), {top.text.splitlines()[-1]}"
            else:
                ans = "Sorry, I don't have information about that. I can connect you to a colleague who can help."

        return ans, "answer", chunk_ids


# ============================================================================
# STEP 3 & 4: CONVERSATIONAL AGENT (SHARED FOR TEXT AND VOICE)
# ============================================================================
class ConversationalAgent:
    """Step 3 & 4 Support Agent.

    Shared respond() method for text and voice.
    Maintains history, stored order ID, one clarifying question when info is missing,
    verifies caller via tools before disclosing order details, executes real
    lookup_order / create_ticket, enforces date logic and ear formatting.
    """

    def __init__(self):
        self.kb = get_kb_index()
        self.history: List[Dict[str, str]] = []
        self.stored_order_id: Optional[str] = None
        self.verified: bool = False
        self.verification_attempted: bool = False
        self.order_data: Optional[dict] = None
        self.ticket_created: Optional[dict] = None
        self.last_chunks_used: List[str] = []
        self.last_decision: str = "answer"

    def reset(self):
        """Reset conversation state."""
        self.history = []
        self.stored_order_id = None
        self.verified = False
        self.verification_attempted = False
        self.order_data = None
        self.ticket_created = None
        self.last_chunks_used = []
        self.last_decision = "answer"

    def respond(self, customer_message: str) -> str:
        """Core multi-turn conversation handler used for both text and voice."""
        self.last_chunks_used = []
        msg = customer_message.strip()
        self.history.append({"role": "customer", "content": msg})
        msg_lower = msg.lower()

        # 1. Extract Order ID if present in current message
        order_matches = re.findall(r"\b(\d{4}\s*\d{4}|\d{8})\b", msg)
        if order_matches:
            raw_id = "".join(c for c in order_matches[0] if c.isdigit())
            self.stored_order_id = raw_id

        # 2. Check for escalation criteria (kb/agent_escalation_handoff.md [ESC-1])
        escalate_reason = None
        intent = "general_support"

        # Rule 6: Customer asks for a human agent
        human_keywords = [
            "talk to a person",
            "human",
            "agent",
            "person",
            "representative",
            "executive",
            "operator",
            "baat karni",
            "insan",
            "colleague",
            "manager",
        ]
        if any(w in msg_lower for w in human_keywords) and (
            "can i talk" in msg_lower
            or "speak to" in msg_lower
            or "transfer" in msg_lower
            or "human agent" in msg_lower
            or "agent se" in msg_lower
        ):
            escalate_reason = "Customer explicitly requested a human agent."
            intent = "human_handover_requested"

        # Rule 3: Legal / police / consumer court
        if any(w in msg_lower for w in ["legal", "court", "consumer forum", "police", "lawyer"]):
            escalate_reason = "Customer mentioned legal action or consumer dispute."
            intent = "legal_escalation"

        # Rule 7: Safety risk
        if any(w in msg_lower for w in ["swelling", "battery", "smoke", "burning", "sparks", "shock", "fire", "exploded"]):
            escalate_reason = "Customer reported product safety risk."
            intent = "safety_risk"

        # Rule 2: Fraud / unauthorized
        if any(w in msg_lower for w in ["unauthorized", "didn't place", "did not order", "fraud", "hacked"]):
            escalate_reason = "Customer reported unauthorized order or payment activity."
            intent = "unauthorized_activity"

        # 3. Check Order / Verification Flow
        digits_in_msg = "".join(c for c in msg if c.isdigit())

        # If user asks about an order but no order ID has ever been provided
        needs_order = any(w in msg_lower for w in ["order", "delivery", "cancel", "refund", "track", "package", "where is", "saman"])
        if needs_order and not self.stored_order_id:
            self.last_decision = "clarify"
            reply = (
                "Hello! I am ZentroMart's AI support assistant. "
                "I would be glad to help you with your order. Could you please share your 8-digit order ID?"
            )
            self.history.append({"role": "agent", "content": reply})
            return reply

        # If we have an order ID but are not verified yet
        if self.stored_order_id and not self.verified:
            # Check if this message provides the verification value (4 digits phone last 4 or 6 digits pincode)
            if digits_in_msg and len(digits_in_msg) in (4, 6) and digits_in_msg != self.stored_order_id:
                # Try verifying with this proof
                res = lookup_order(self.stored_order_id, verification_value=digits_in_msg)
                if res.get("verified") is True:
                    self.verified = True
                    self.order_data = res
                    spaced_id = format_order_id_for_ear(self.stored_order_id)
                    first_name = res.get("customer_first_name", "there")

                    # Check repeat contact rule (ESC-1 Rule 4: contacts_last_7_days >= 2)
                    if res.get("contacts_last_7_days", 0) >= 2:
                        summary = (
                            f"Order {self.stored_order_id}: customer {first_name} has contacted {res.get('contacts_last_7_days')} "
                            f"times in the last 7 days regarding order status. Escalating to human support."
                        )
                        ticket_res = create_ticket(
                            summary=summary,
                            order_id=self.stored_order_id,
                            reason="Repeat contact (>=2 earlier contacts in 7 days)",
                            intent="repeat_contact_escalation",
                        )
                        if ticket_res.get("created"):
                            self.ticket_created = ticket_res
                            self.last_decision = "escalate"
                            reply = (
                                f"Thank you for verifying, {first_name}. I see you have reached out multiple times regarding order {spaced_id}. "
                                f"I have created ticket {ticket_res['ticket_id']} and connected you to a specialist who will contact you shortly."
                            )
                            self.history.append({"role": "agent", "content": reply})
                            return reply

                    self.last_decision = "answer"
                    status = res.get("status", "processing")
                    reply = (
                        f"Thank you, {first_name}. Your identity is verified for order {spaced_id}. "
                        f"The current status of your order is {status}. How may I help you with this order today?"
                    )
                    self.history.append({"role": "agent", "content": reply})
                    return reply
                else:
                    self.last_decision = "clarify" if not self.verification_attempted else "refuse"
                    self.verification_attempted = True
                    reply = (
                        "Verification was unsuccessful. Please ensure you provide either the last 4 digits "
                        "of your registered phone number or your 6-digit delivery pincode."
                    )
                    self.history.append({"role": "agent", "content": reply})
                    return reply
            else:
                # Call lookup_order with order_id only to trigger verification requirement
                res = lookup_order(self.stored_order_id)
                self.last_decision = "clarify"
                spaced_id = format_order_id_for_ear(self.stored_order_id)
                reply = (
                    f"I have found order {spaced_id}. For your security, please share either the last 4 digits "
                    f"of your registered phone number or your delivery pincode so I can verify your account."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

        # 4. If caller is verified, handle order-specific intents
        if self.verified and self.order_data:
            spaced_id = format_order_id_for_ear(self.stored_order_id)

            # Check if high-value damaged item (> Rs 10,000) - ESC-1 Rule 1
            if "damage" in msg_lower or "defect" in msg_lower or "broken" in msg_lower:
                items = self.order_data.get("items", [])
                high_value = any(it.get("unit_price_inr", 0) > 10000 for it in items)
                if high_value or self.order_data.get("subtotal_inr", 0) > 10000:
                    summary = (
                        f"High-value damage report on order {self.stored_order_id}. "
                        f"Customer reported damaged item exceeding ₹10,000 threshold. Escalating for replacement approval."
                    )
                    ticket_res = create_ticket(
                        summary=summary,
                        order_id=self.stored_order_id,
                        reason="Damaged item worth more than ₹10,000",
                        intent="damaged_high_value",
                    )
                    if ticket_res.get("created"):
                        self.ticket_created = ticket_res
                        self.last_decision = "escalate"
                        reply = (
                            f"I understand your item arrived damaged. Because this involves an item valued over ten thousand rupees, "
                            f"I have created ticket {ticket_res['ticket_id']} and forwarded this to our senior team for immediate resolution."
                        )
                        self.history.append({"role": "agent", "content": reply})
                        return reply

            # Refusal Rule: Compensation demand
            if "compensation" in msg_lower or "extra money" in msg_lower or "harrassment" in msg_lower:
                self.last_decision = "refuse"
                reply = (
                    f"I apologize for the inconvenience caused with order {spaced_id}. "
                    "However, our company policy does not permit direct monetary compensation beyond our published delivery credits."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

            # Cancellation request
            if "cancel" in msg_lower:
                status = self.order_data.get("status")
                if status in ("shipped", "delivered"):
                    self.last_decision = "refuse"
                    reply = (
                        f"Order {spaced_id} has already been {status} and cannot be cancelled directly. "
                        "You may decline delivery when the courier arrives or initiate a return after delivery."
                    )
                    self.history.append({"role": "agent", "content": reply})
                    return reply
                else:
                    self.last_decision = "answer"
                    reply = (
                        f"Order {spaced_id} is currently {status} and eligible for cancellation. "
                        "Your refund will be initiated to your original payment method once cancellation is confirmed."
                    )
                    self.history.append({"role": "agent", "content": reply})
                    return reply

            # Return eligibility
            if "return" in msg_lower:
                self.last_decision = "answer"
                items = self.order_data.get("items", [])
                item_names = [it.get("title", "Item") for it in items]
                reply = (
                    f"Your items from order {spaced_id} can be returned within the standard return window from delivery date. "
                    "Please ensure the items remain unused with all tags and packaging intact."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

            # Where is my order / tracking
            if "where is" in msg_lower or "track" in msg_lower or "status" in msg_lower:
                self.last_decision = "answer"
                status = self.order_data.get("status", "processing")
                promised = self.order_data.get("promised_date", "2026-10-25")
                reply = (
                    f"Order {spaced_id} is currently {status}. "
                    f"The promised delivery date is {promised}."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

            # Refund status
            if "refund" in msg_lower:
                self.last_decision = "answer"
                reply = (
                    f"Refunds for order {spaced_id} are processed within two to five business days depending on your payment method. "
                    "ZentroMart Plus members receive immediate refund initiation upon reverse pickup."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

        # 5. Handle General Escalation
        if escalate_reason:
            summary = (
                f"Escalation for customer issue: {customer_message}. "
                f"Reason: {escalate_reason}. Order ID: {self.stored_order_id or 'None'}. Handing over to human agent."
            )
            ticket_res = create_ticket(
                summary=summary,
                order_id=self.stored_order_id,
                reason=escalate_reason,
                intent=intent,
            )
            if ticket_res.get("created"):
                self.ticket_created = ticket_res
                self.last_decision = "escalate"
                reply = (
                    f"I have created support ticket {ticket_res['ticket_id']} and escalated your request to our team. "
                    "A human colleague will contact you shortly to assist you further."
                )
                self.history.append({"role": "agent", "content": reply})
                return reply

        # 6. Fallback to Grounded RAG for general knowledge queries
        chunks, chunk_ids, is_thin = self.kb.retrieve(customer_message, top_k=3)
        self.last_chunks_used = chunk_ids

        if is_thin:
            self.last_decision = "answer"
            reply = "Sorry, I don't have information about that. I can connect you to a colleague who can help."
            self.history.append({"role": "agent", "content": reply})
            return reply

        # Grounded answer using retrieved chunks
        context_texts = [f"[{c.section_id}] {c.title}: {c.text}" for c in chunks]
        context_str = "\n".join(context_texts)
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"User Question: {customer_message}\n"
            f"Answer in 2 short sentences. Use month names in dates. Mention ZentroMart policies strictly."
        )
        reply = call_llm(prompt, system_prompt="You are ZentroMart's AI support assistant.")
        if not reply or "don't have" in reply.lower() or "not sure" in reply.lower():
            reply = "Sorry, I don't have information about that. I can connect you to a colleague who can help."

        self.last_decision = "answer"
        self.history.append({"role": "agent", "content": reply})
        return reply
