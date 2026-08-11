F5 调试快照：sql_generation_prompt 的一次实际值，仅供理解 Prompt 拼装结果。
根据问题生成MySQL SELECT语句。只输出SQL，不要解释。

当前用户ID：d1e88d66-cc87-40c7-bbe3-2dff2d093b41
今天日期：2026-08-03
当前用户部门ID：

常用日期范围（直接使用，不要自行计算）：
- 本月：2026-08-01 至 2026-08-03
- 本周：2026-08-03 至 2026-08-03
- 上周：2026-07-27 至 2026-08-02

表结构：
sys_user(id PK, entity_name 员工姓名, dept_id FK→org_dept.id)
workhour(id PK, member_id FK→sys_user.id, workhour_date datetime,
         workhour decimal 小时数, project_id FK→project_info.id,
         work_content 工作内容, is_deleted 0未删/1已删)

表关联规则：
- workhour.member_id = sys_user.id（人员）
- workhour.project_id = project_info.id（项目）
- sys_user.org_id = org_dept.ext_field_2（部门，用于获取部门名称：
  org_dept.dept_name；注意是 org_id 而非 dept_id）
- project_member.member_id = sys_user.id（项目成员）
- workhour_attendance.member_id = sys_user.id（考勤/加班，关键字段：
  overtime_hours加班时长、check_in_time上班打卡、
  check_out_time下班打卡、work_date日期）

【部门层级规则】org_dept 表中 ext_field_2 是父部门 ID：
- 二级部门：id = ext_field_2（自己指向自己，即有子部门归在它下面的部门）
- 三级子部门：id != ext_field_2（有明确父部门的班组/小组）
- 用户问“各部门”“部门占比”“部门排名”时，只统计二级部门
  （WHERE od.id = od.ext_field_2），不展开到三级子部门，避免图表过碎。
- 关联写法：JOIN org_dept od ON su.org_id = od.ext_field_2
  WHERE od.id = od.ext_field_2

【漏填工时专用查询模板】查询用户在某日期区间内哪些工作日没有填工时：
SELECT DATE(wc.date_value) AS 日期, wc.work_hour AS 应填工时
FROM work_calendar wc
LEFT JOIN workhour wh
  ON DATE(wc.date_value)=DATE(wh.workhour_date)
 AND wh.member_id='d1e88d66-cc87-40c7-bbe3-2dff2d093b41'
WHERE wc.is_work_day='1'
  AND DATE(wc.date_value) BETWEEN '起始日期' AND '结束日期'
  AND wh.id IS NULL
ORDER BY wc.date_value LIMIT 100
说明：work_calendar.is_work_day='1'（字符串）；work_calendar.work_hour
是该天应填工时；不要 JOIN sys_user 或 org_dept（漏填时 wh 记录不存在，
JOIN 结果全为 NULL）。

【加班时长专用查询模板】查询个人加班数据（workhour_attendance 表）：
SELECT wa.work_date AS 日期, wa.overtime_hours AS 加班时长,
       wa.overtime_type AS 加班类型
FROM workhour_attendance wa
WHERE wa.member_id='d1e88d66-cc87-40c7-bbe3-2dff2d093b41'
  AND wa.work_date BETWEEN '起始日期' AND '结束日期'
  AND wa.overtime_hours > 0
ORDER BY wa.work_date LIMIT 100
说明：overtime_hours 是 decimal 类型；汇总加班总时长用
SUM(wa.overtime_hours)；member_id 对应 sys_user.id。

【部门加班统计模板】当问题含“部门”“本部门”“排名”时，
查部门内所有人的加班排名：
SELECT su.entity_name AS 姓名,
       SUM(wa.overtime_hours) AS 加班总时长
FROM workhour_attendance wa
JOIN sys_user su ON wa.member_id = su.id
JOIN org_dept od ON su.org_id = od.ext_field_2
WHERE od.ext_field_2 = ''
  AND wa.work_date BETWEEN '起始日期' AND '结束日期'
  AND wa.overtime_hours > 0
GROUP BY su.id, su.entity_name
ORDER BY SUM(wa.overtime_hours) DESC LIMIT 100
说明：od.ext_field_2 是部门ID，与 sys_user.org_id 关联；必须用
JOIN sys_user + org_dept 扩展范围，不得仅限制当前 member_id。

权限约束：数据范围限制（自动注入 WHERE 条件）：
  workhour.member_id IN ('d1e88d66-cc87-40c7-bbe3-2dff2d093b41')

生成规则：
1. 只生成 SELECT 语句，禁止 INSERT/UPDATE/DELETE
2. workhour.workhour_date 是 datetime 类型，日期比较必须用 DATE() 函数
3. 人员姓名用 sys_user.entity_name 字段
4. 部门名称用 org_dept.dept_name 字段，关联条件为
   sys_user.org_id = org_dept.ext_field_2
5. 所有列名必须使用中文别名（AS 中文名）
6. 结果必须加 LIMIT（默认 LIMIT 100）
7. 涉及漏填工时时，必须使用上方漏填工时专用查询模板
8. 涉及部门范围时，必须使用部门加班统计模板

问题：上个月加班时长最多的三个人，分别加了多少小时？