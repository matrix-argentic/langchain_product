import operator
import os
from pathlib import Path
from pydoc import doc
from typing import Annotated, List, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime

# 准备环境变量
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "sql_assistant"
embedding_model = os.getenv("EMBEDDING_MODEL")
hf_home = os.getenv("HF_HOME")
model_cache_dir = hf_home + "\\hub"

# 模型相关
api_key = os.getenv("DASHSCOPE_API_KEY", "")
api_base = os.getenv("DASHSCOPE_API_BASE", "")
model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")

# 路径相关
root_dir = Path(__file__).parents[2]
data_dir = root_dir / "data"


def get_chat_model():
    from langchain_qwq import ChatQwen

    model = ChatQwen(
        model=model_name,
        max_tokens=3_000,
        timeout=None,
        max_retries=2,
        api_key=api_key,
        base_url=api_base,
        enable_thinking=False,  # 关闭深度思考
        # other params...
    )
    return model


#### 技能定义 ####


class Skill(TypedDict):
    """一种可逐步向智能体揭示的技能。"""

    name: str  # 技能的唯一标识符
    description: str  # 在系统提示中显示的 1-2 句描述
    content: str  # 包含详细说明的完整技能内容


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "用于销售数据分析的数据库架构与业务逻辑，包含客户、订单与收入相关数据。",
        "content": """# 销售分析数据库架构

## 数据表

### customers（客户表）
- customer_id（主键）
- name（姓名）
- email（邮箱）
- signup_date（注册日期）
- status（状态：active/inactive）
- customer_tier（客户等级：bronze/silver/gold/platinum）

### orders（订单表）
- order_id（主键）
- customer_id（外键 -> customers表）
- order_date（订单日期）
- status（状态：pending/completed/cancelled/refunded）
- total_amount（总金额）
- sales_region（销售区域：north/south/east/west）

### order_items（订单明细表）
- item_id（主键）
- order_id（外键 -> orders表）
- product_id（商品ID）
- quantity（数量）
- unit_price（单价）
- discount_percent（折扣比例）

## 业务逻辑

**活跃客户**：status = 'active' 且 signup_date <= CURRENT_DATE - INTERVAL '90 days'

**收入计算**：仅统计 status = 'completed' 的订单。使用 orders 表中的 total_amount 字段，该字段已包含折扣计算。

**客户终身价值（CLV）**：客户所有已完成订单的金额总和。

**高价值订单**：total_amount > 1000 的订单。

## 查询示例

-- 获取最近一季度按收入排序的前10位客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",
        "description": "用于库存跟踪的数据库架构与业务逻辑，包含商品、仓库与库存水平信息。",
        "content": """# 库存管理数据库架构

## 数据表

### products（商品表）
- product_id（主键）
- product_name（商品名称）
- sku（库存单位编码）
- category（类别）
- unit_cost（单位成本）
- reorder_point（补货点：触发重新订货的最低库存水平）
- discontinued（是否下架，布尔型）

### warehouses（仓库表）
- warehouse_id（主键）
- warehouse_name（仓库名称）
- location（位置）
- capacity（容量）

### inventory（库存表）
- inventory_id（主键）
- product_id（外键 -> products表）
- warehouse_id（外键 -> warehouses表）
- quantity_on_hand（当前库存数量）
- last_updated（最后更新时间）

### stock_movements（库存移动记录表）
- movement_id（主键）
- product_id（外键 -> products表）
- warehouse_id（外键 -> warehouses表）
- movement_type（移动类型：inbound/outbound/transfer/adjustment）
- quantity（数量：入库为正，出库为负）
- movement_date（移动日期）
- reference_number（参考编号）

## 业务逻辑

**可用库存**：inventory 表中 quantity_on_hand > 0 的记录

**需补货商品**：所有仓库中总 quantity_on_hand 小于等于该商品 reorder_point 的商品

**仅限活跃商品**：除非专门分析已下架商品，否则排除 discontinued = true 的商品

**库存估值**：每个商品的 quantity_on_hand * unit_cost

## 查询示例

-- 查找所有仓库中库存低于补货点的商品
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]


from langchain.tools import tool


@tool
def load_skill(skill_name: str) -> str:
    """将特定技能的完整内容加载到智能体的上下文中。

    当您需要处理特定类型请求的详细信息时使用此工具。这将为您提供该技能领域的详细说明、政策与指南。

    参数:
        skill_name: 要加载的技能名称（例如："expense_reporting"、"travel_booking"）
    """
    # 查找并返回请求的技能
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"已加载技能：{skill_name}\n\n{skill['content']}"

    # 未找到对应技能
    available = ", ".join(s["name"] for s in SKILLS)
    return f"技能 '{skill_name}' 不存在。可用技能列表：{available}"


from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable


class SkillMiddleware(AgentMiddleware):
    """将技能描述注入系统提示词的中间件。"""

    # 将 load_skill 工具注册为类变量
    tools = [load_skill]

    def __init__(self):
        """初始化，并根据 SKILLS 列表生成技能提示内容。"""
        # 基于 SKILLS 列表构建技能提示
        skills_list = []
        for skill in SKILLS:
            skills_list.append(f"- **{skill['name']}**：{skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步调用：将技能描述注入系统提示词。"""
        # 构建技能附加信息
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "当您需要详细了解如何处理特定类型的请求时，请使用 load_skill 工具。"
        )

        # 追加到系统消息内容块
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


def get_sql_agent(model):
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    # Create the agent with skill support
    agent = create_agent(
        model,
        system_prompt=("您是一个 SQL 查询助手，帮助用户对业务数据库编写查询语句。"),
        middleware=[SkillMiddleware()],
        checkpointer=InMemorySaver(),
    )
    return agent


# uv run -m src.multi_agent.sql_assistant
if __name__ == "__main__":
    from langchain_core.utils.uuid import uuid7

    model = get_chat_model()
    agent = get_sql_agent(model=model)
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    # Ask for a SQL query
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a SQL query to find all customers "
                        "who made orders over $1000 in the last month"
                    ),
                }
            ]
        },
        config,
    )
    for message in result["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")
