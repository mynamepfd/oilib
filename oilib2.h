#pragma once

/*
oilib2的结构如下

namespace lib2 {
	namespace pABC {

	}
}

其中ABC表示该程序在视频教程的第A章第B节的第C个视频中
*/

namespace lib2 {
	// 分治法 - P1314. 聪明的质检员
	namespace p122 {
		void read_case();
		void solve();
	}

	// 分治法 - P2678. 跳石头
	namespace p123 {
		void read_case();
		void solve();
	}

	// 分治法 - P1083. 借教室
	namespace p124 {
		void read_case();
		void solve();
	}

	// 倍增法 - P1081. 开车旅行
	namespace p125 {
		void read_case();
		void solve();
	}

	// 分治法 - CF553D Nudist Beach
	namespace p126 {
		void read_case();
		void solve();
	}

	// CF359D Pair of Numbers
	namespace p127 {
		void read_case();
		void solve();
	}

	// CF429D Tricky Function
	namespace p128 {
		void read_case();
		void solve();
	}

	// CF414C Mashmokh and Reverse Operation
	namespace p129 {
		void read_case();
		void solve();
	}

	// CF702E Analysis of Pathes in Functional Graph
	namespace p1210 {
		void read_case();
		void solve();
	}

	// 贪心 - P1969. 积木大赛
	namespace p131 {
		void read_case();
		void solve();
	}

	// 贪心 - P1080. 国王游戏
	namespace p132 {
		void read_case();
		void solve();
	}

	// 贪心 - P1966. 火柴排队
	namespace p133 {
		void read_case();
		void solve();
	}

	// 贪心 - P1315. 观光公交
	namespace p134 {
		void read_case();
		void solve();
	}

	// 贪心 - CF496E Distributing Parts
	namespace p135 {
		void read_case();
		void solve();
	}

	// 贪心 - CF767E Change Free
	namespace p136 {
		void read_case();
		void solve();
	}

	// 贪心 - CF883J Renovation
	namespace p137 {
		void read_case();
		void solve();
	}

	// 贪心 - CF639E Bear and Paradox
	namespace p138 {
		void read_case();
		void solve();
	}

	// 贪心 - HDU 6299. Balanced Sequence
	namespace p139 {
		// TODO
	}

	// 贪心 - HDU 5303. Delicious Apples
	namespace p1310 {
		// TODO
	}

	// 搜索 - POJ 1011. Sticks
	namespace p141 {
		void read_case();
		void solve();
	}

	// 搜索 - P2668. 斗地主
	namespace p142 {
		extern int n;
		void read_case();
		void solve();
	}

	// 搜索 - Mayan游戏
	namespace p143 {
		void read_case();
		void solve();
	}

	// 搜索 - POJ 3009. Curling 2.0
	namespace p144 {
		void read_case();
		void solve();
	}

	// 搜索 - POJ 1190. 生日蛋糕
	namespace p145 {
		void read_case();
		void solve();
	}

	// 搜索 - HDU 6171. Admiral
	namespace p146 {
		void read_case();
		void solve();
	}

	// 基本数据结构 - 列队
	namespace p151 {
		void read_case();
		void solve();
	}

	// 基本数据结构 - P1631. 有序表的最小和
	namespace p152_part1 {
		void read_case();
		void solve();
	}

	// 基本数据结构 - UVA136. 丑数
	namespace p152_part2 {
		void read_case();
		void solve();
	}

	// 基本数据结构 - 轮廓线
	namespace p152_part3 {
		// TODO
	}

	// 基本数据结构 - POJ 3784. Running Median
	namespace p153_part1 {
		// TODO
	}

	// 基本数据结构 - 线段
	namespace p153_part2 {
		// TODO
	}

	// 随机化 - P1970. 花匠
	namespace p162 {
		void read_case();
		void solve();
	}

	// 随机化 - CF869E. The Untended Antiquity
	namespace p163 {
		void read_case();
		void solve();
	}

	// 分块法 - CF785E. Anoton and Permutation
	namespace p171 {
		void read_case();
		void solve();
	}

	// 分块法 - P4135. 作诗
	namespace p172 {
		void read_case();
		void solve();
	}

	// 逛公园
	namespace p181 {
		// TODO
	}

	// 天天爱跑步
	namespace p182 {
		// TODO
	}

	// 蚯蚓
	namespace p183 {
		// TODO
	}

	// 运输计划
	namespace p184 {
		// TODO
	}

	// 疫情控制
	namespace p185 {
		// TODO
	}

	// 华容道
	namespace p186 {
		// TODO
	}

	// 动态规划 - POJ3176. 数塔
	namespace p223_part1 {
		// TODO
	}

	// 动态规划 - POJ1088. 滑雪
	namespace p223_part2 {
		// TODO
	}

	// 动态规划 - POJ3481. Computers
	namespace p231 {
		// TODO
	}

	// 动态规划 - 合并果子
	namespace p232_part1 {
		// TODO
	}

	// 动态规划 - 括号匹配
	namespace p232_part2 {
		// TODO
	}

	// 动态规划 - POJ 1159. Palindrome
	namespace p233_part1 {
		// TODO
	}

	// 动态规划 - UVA 10617. Again Palindromes
	namespace p233_part2 {
		// TODO
	}

	// 动态规划 - HDU 2476. String Painter
	namespace p234 {
		// TODO
	}

	// 动态规划 - HDU 1421. 搬寝室
	namespace p235 {
		// TODO
	}

	// 动态规划 - BZOJ 1786. 配对
	namespace p236 {
		// TODO
	}

	// 背包问题 - 01背包
	namespace p241_part1 {
		// TODO
	}

	// 背包问题 - 完全背包
	namespace p241_part2 {
		// TODO
	}

	// 背包问题 - 多重背包
	namespace p242_part1 {
		// TODO
	}

	// 背包问题 - 多重背包的二进制优化
	namespace p242_part2 {
		// TODO
	}

	// 背包问题 - 多重背包的单调队列优化
	namespace p243 {
		// TODO
	}

	// 背包问题 - CF 366C. Dima and Salad
	namespace p244 {
		// TODO
	}

	// 背包问题 - CF 864C. Fire
	namespace p245_part1 {
		// TODO
	}

	// 背包问题 - POJ 1742. Coins
	namespace p245_part2 {
		// TODO
	}

	// 背包问题 - CF 755F. PolandBall and Gifts
	namespace p246 {
		// TODO
	}

	// 背包问题 - 飞扬的小鸟
	namespace p247 {
		// TODO
	}

	// 数位DP - BZOJ 1799. 同类分布
	namespace p251 {
		// TODO
	}

	// 数位DP - HDU 3709. Balanced Number
	namespace p252 {
		// TODO
	}

	// 数位DP - HDU 4507.
	namespace p253 {
		// TODO
	}

	// 数位DP - CF 55D. Beautiful Numbers
	namespace p254 {
		// TODO
	}

	// 数位DP - Codechef. FAVNUM
	namespace p255 {
		// TODO
	}

	// 树上DP - HDU 1561. The more, the better
	namespace p262 {
		// TODO
	}

	// 树上DP - HDU 5593. ZYB's Tree
	namespace p263 {
		// TODO
	}

	// 树上DP - POJ 2486. Apple Tree
	namespace p264 {
		// TODO
	}

	// 树上DP - CF 960E. Alternating Tree
	namespace p265 {
		// TODO
	}

	// 状态压缩DP - HDU 4628. Pieces
	namespace p272 {
		// TODO
	}

	// 状态压缩DP - HDU 6149. Valley Number II
	namespace p273 {
		// TODO
	}

	// 状态压缩DP - 愤怒的小鸟
	namespace p274 {
		// TODO
	}

	// 状态压缩DP - 宝藏
	namespace p275 {
		// TODO
	}

	// DP优化 - 滚动数组优化 - 子串
	namespace p281 {
		// TODO
	}

	// DP优化 - 矩阵乘法优化 - POJ 3734. Blocks
	namespace p282 {
		// TODO
	}

	// DP优化 - 前缀数组优化 - BZOJ 1705. Telephone Wire
	namespace p283 {
		// TODO
	}

	// DP优化 - 换教室
	namespace p284 {
		// TODO
	}

	// DP优化 - HDU 2294. Pendant
	namespace p285 {
		// TODO
	}

	// DP优化 - CF 985E. Pencils and boxes
	namespace p286 {
		// TODO
	}

	// 单调性优化 - 斜率优化 - HDU 3507. Print Article
	namespace p291 {
		// TODO
	}

	// 图算法 - dijkstra算法 - POJ 1724. Roads
	namespace p326 {
		// TODO
	}

	// 图算法 - bellman-ford算法 - POJ 1556. The doors
	namespace p334 {
		// TODO
	}

	// 图算法 - bellman-ford算法 - POJ 2240. 套利
	namespace p335 {
		// TODO
	}

	// 图算法 - spfa算法 - POJ 3013. Big Christmas Tree
	namespace p344 {
		// TODO
	}

	// 图算法 - spfa算法 - HDU 4725. The shortest path in Nya graph
	namespace p345 {
		// TODO
	}

	// 图算法 - floyd算法 - POJ 3660. Cow Contest
	namespace p353 {
		// TODO
	}

	// 图算法 - floyd算法 - POJ 3613. Cow Relays
	namespace p354 {
		// TODO
	}

	// 图算法 - 差分约束 - ZOJ 2770. 火烧连营
	namespace p362 {
		// TODO
	}

	// 图算法 - 差分约束 - POJ 1201. Intervals
	namespace p363 {
		// TODO
	}

	// 图算法 - 算法总结 - HDU 6166. Senior Pan
	namespace p372 {
		// TODO
	}

	// 图算法 - 算法总结 - HDU 4370. 0 or 1
	namespace p373 {
		// TODO
	}

	// 线段树 - 单点修改
	namespace p523 {
		// TODO
	}

	// 线段树 - 区间修改
	namespace p525 {
		// TODO
	}
}
