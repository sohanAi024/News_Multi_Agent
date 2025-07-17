from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def message_to_dict(message):
    """Convert LangChain message to dictionary"""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        return {"role": message.type, "content": message.content}
    return message

def dict_to_message(msg_dict):
    """Convert dictionary to LangChain message"""
    role = msg_dict.get("role")
    content = msg_dict.get("content")
    if role == "user":
        return HumanMessage(content=content)
    elif role in ("assistant", "ai"):
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    return msg_dict