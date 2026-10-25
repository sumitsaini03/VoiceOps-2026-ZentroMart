# VoiceOps 2026: ZentroMart Support Agent - 2-Minute Demo Video Script

**Target Duration**: 1 minute 45 seconds to 2 minutes  
**Format**: Spoken dialogue recorded with screen showing CLI / terminal interaction and audio playback.

---

## Demo Overview & Turn Sequence

This script demonstrates the three core capabilities required by the VoiceOps rubric:
1. **Document-Grounded Answer with Follow-Up** (Return window for electronics & festive sale)
2. **Identity Verification & Order Status Check** (Gated order lookup using phone last 4)
3. **Graceful Refusal & Escalation Handover** (Handling policy-prohibited compensation & ticket generation)

---

## Turn 1: Grounded Answer from Documents (0:00 – 0:35)

- **Customer (Spoken)**:  
  *"Hi, I bought a smartphone last week. What is the return window for electronics, and does the festive sale apply to it?"*

- **Agent (Voice Reply via Gnani 'Kaveri' / Text)**:  
  *"Hello! I am ZentroMart's AI support assistant. For electronics and accessories, the return window is seven calendar days from delivery under our current policy. The 30-day festive sale return window applies only to clothing, footwear, and bags, so normal return windows apply to your smartphone."*

- **Internal Verification**:
  - Retrieved Chunks: `[RET-2]`, `[RET-3]`, `[RET-8]`
  - Decision: `answer`
  - Precedence: `pol_returns_v4.md` (current) used over `pol_returns_v3_superseded.md`.

---

## Turn 2: Follow-Up & Identity Verification Gate (0:35 – 1:05)

- **Customer (Spoken)**:  
  *"Can you check the delivery status of my order? The order number is 8978 2837."*

- **Agent (Voice Reply)**:  
  *"I have found order 8978 2837. For your security, please share either the last 4 digits of your registered phone number or your delivery pincode so I can verify your account."*

- **Internal Verification**:
  - Tool Call: `lookup_order('89782837')` returns `{'verification_required': True}`
  - Identity Gate: Zero order details disclosed before verification.
  - Decision: `clarify`

---

## Turn 3: Successful Verification & Order Disclosure (1:05 – 1:25)

- **Customer (Spoken)**:  
  *"The last four digits of my phone number are 4533."*

- **Agent (Voice Reply)**:  
  *"Thank you, Kabir. Your identity is verified for order 8978 2837. The current status of your order is delivered on 15 October 2026. How may I help you with this order today?"*

- **Internal Verification**:
  - Tool Call: `lookup_order('89782837', verification_value='4533')` returns `{'verified': True, 'found': True}`
  - Decision: `answer`

---

## Turn 4: Policy Refusal & Human Handover Escalation (1:25 – 1:55)

- **Customer (Spoken)**:  
  *"The 43-inch TV in this order arrived damaged, and I demand ₹500 cash compensation immediately for my trouble."*

- **Agent (Voice Reply)**:  
  *"I apologize for the trouble with your television. However, company policy does not permit direct cash compensation. Because your damaged item is valued over ten thousand rupees, I have created support ticket TKT92bcd01ef62e4714 and escalated this to our senior specialist team, who will contact you shortly."*

- **Internal Verification**:
  - Refusal: Politely declines compensation demand.
  - Tool Call: `create_ticket(summary="High-value damage report on order 89782837. Customer reported damaged TV exceeding ₹10,000 threshold. Escalating for replacement approval.", order_id='89782837', reason="Damaged item worth more than ₹10,000", intent="damaged_high_value")`
  - Ticket Confirmation: Confirmed only because `created=True`.
  - Decision: `escalate`

---

## Tips for Recording the Video
1. Open a terminal and run `python run.py --interactive` or play pre-recorded audio clips with `python run.py sample_audio/`.
2. Speak clearly into the microphone or let Gnani's TTS voice Kaveri speak the agent's replies.
3. Keep total video duration under 2 minutes.
