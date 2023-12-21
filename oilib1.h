#pragma once

// 隐式图的dfs - 部分和问题
namespace p30 {
	const int MAX_N = 20;
	
	extern int a[MAX_N];
	extern int n, k;

	void solve(); 
}

// 八方向dfs - Lake Counting (POJ 2386)
namespace p32 {
	const int MAX_N = 20;
	const int MAX_M = 20;
	
	extern char field[MAX_N][MAX_M];
	extern int n, m;

	void solve();
}

// bfs - 迷宫最短路径
namespace p34 {

	const int INF = 0x3f3f3f3f;
	const int MAXN = 100;
	const int MAXM = 100;

	extern char maze[MAXN][MAXM + 1];
	extern int N, M;
	extern int sx, sy;
	extern int gx, gy;
	extern int d[MAXN][MAXM];

	void solve();
}

// DP - 01背包
namespace p51 {

	const int MAX_N = 100;
	const int MAX_W = 1000;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - 最长公共子序列
namespace p56 {

	const int maxn = 1000;
	const int maxm = 1000;

	extern int n, m;
	extern string s, t;

	void solve();
}

// DP - 完全背包
namespace p57 {

	const int MAX_N = 100;
	const int MAX_W = 1000;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - 01背包变形
namespace p60 {

	const int MAX_N = 100;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - 多重部分和问题
namespace p62 {
	const int MAX_N = 100;
	
	extern int n, K;
	extern int a[MAX_N], m[MAX_N];

	void solve();
}

// DP - 多重部分和问题优化
namespace p63 {
	const int MAX_N = 100;

	extern int n, K;
	extern int a[MAX_N], m[MAX_N];

	void solve();
}

// DP - 最长上升子序列问题
namespace p64 {
	const int MAX_N = 1000;

	extern int n;
	extern int a[MAX_N];

	void solve();
}

// DP - 最长上升子序列问题优化
namespace p65 {
	const int MAX_N = 1000;

	extern int n;
	extern int a[MAX_N];

	void solve();
}

// DP - 划分数
namespace p66 {

	const int MAX_N = 1000;
	const int MAX_M = 1000;

	extern int n, m, M;

	void solve();
}

// DP - 多重集组合数
namespace p68 {

	const int MAX_N = 1000;
	const int MAX_M = 1000;

	extern int n, m, a[MAX_N], M;

	void solve();
}

/*
并查集
*/
namespace p84 {

	const int MAX_N = 1;

	extern int par[MAX_N], rank[MAX_N];
	void init(int n);
	int find(int u);
	void unite(int u, int v);
	bool same(int u, int v);
}


// 并查集 - POJ 1182. 食物链
namespace p88 {

	const int MAX_K = 100000;

	extern int N, K;
	extern int T[MAX_K], X[MAX_K], Y[MAX_K];//第i条信息的类型,X_i,y_i

	void solve();
}

// 图算法 - 二分图判定
namespace p97 {
	const int MAX_V = 1000;

	extern VI G[MAX_V];	//图
	extern int V;		//顶点数
	
	void solve();
}

// 图算法 - Bellmsn-Ford算法
namespace p100 {
	const int INF = 1000000;
	
	struct edge { int from, to, cost; };
	extern vector<edge> es;	//图
	extern int V,E;			//顶点数

	void shortest_path(int s);
	bool find_negative_loop();
}

// 图算法 - 邻接矩阵Dijkstra算法
namespace p101 {
	const int INF = 1000000;
	const int MAX_V = 100;

	extern int cost[MAX_V][MAX_V];	//图
	extern int V;					//顶点数

	void dijkstra(int s);
}

// 图算法 - 邻接表Dijkstra算法
namespace p102 {
	const int INF = 1000000;
	const int MAX_V = 1000;

	struct edge { int to, cost; };
	extern vector<edge> G[MAX_V];	//图
	extern int V;			//顶点数

	void dijkstra(int s);
}

// 图算法 - Floyd算法
namespace p103 {
	const int INF = 1000000;
	const int MAX_V = 1000;

	extern int d[MAX_V][MAX_V];	//图
	extern int V;			//顶点数

	void floyd(int s);
}

// 图算法 - 路径还原
namespace p104 {
	const int INF = 1000000;
	const int MAXV = 100;

	extern int cost[MAXV][MAXV];	//图
	extern int V;					//顶点数

	void dijkstra(int s);
	vector<int> get_path(int t);
}

// 图算法 - 最小生成树Prim算法
namespace p105 {
	const int INF = 1000000;
	const int MAXV = 100;

	extern int cost[MAXV][MAXV];	//图
	extern int V;					//顶点数

	void prim(int s);
}

// 图算法 - 最小生成树Kruskal算法
namespace p107 {

	struct edge { int u, v, cost; };
	extern vector<edge> es;		//图
	extern int V, E;			//顶点数,边数

	int kruskal(int s);
}

// 欧几里得算法
namespace p113 {
	int gcd(int a, int b); // 求(a,b)
}

// 扩展欧几里得算法
namespace p115 {
	int extgcd(int a, int b, int& x, int& y); // 求x,y使得ax+by=1
}

// 素数判定
namespace p117 {
	bool is_prime(int n); // 素数判定
	vector<int> divisor(int n); // 约数枚举
	map<int, int> prime_factor(int n); // 质因数分解
}

// 数学算法 - 埃拉托色尼筛法
namespace p118 {

	const int MAX_N = 1000000;
	extern int prime[MAX_N];
	extern int is_prime[MAX_N+1];

	int sieve(int n);
}

// 数学算法 - 区间筛法
namespace p120 {

	const int MAX_L = 1000000;
	const int MAX_SQRT_B = 1000000;

	extern bool is_prime[MAX_L];
	extern bool is_prime_small[MAX_SQRT_B + 1];

	void segment_sieve(LL a, LL b);
}

// 数学算法 - 快速幂
namespace p122 {

	LL mod_pow(LL x, LL n, LL m);
}

// lower_bound的实现
namespace p138 {
	const int MAX_N = 1000000;
	extern int n, k, a[MAX_N];

	void solve();
}

// 二分答案 - Cable master (POJ 1064)
namespace p140 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int N, K;
	extern double L[MAX_N];

	void solve();
}

// 最大化最小值 - Aggressive Cows (POJ 2456)
namespace p142 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int N, M, x[MAX_N];

	void solve();
}

// 最大化平均值
namespace p143 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int n, k, w[MAX_N], v[MAX_N];

	void solve();
}

/*
尺取法 -  POJ 3061. Subsequence
给定长度为n的数列a，整数S。找出最短的总和不小于S的连续子序列。
*/
namespace p146 {

	const int MAX_N = 100000;

	extern int n, S, a[MAX_N];

	void solve();
}

/*
尺取法 -  POJ 3320. Jessica's Reading Problem
*/
namespace p149 {

	const int MAX_P = 1000000;
	extern int P;
	extern int a[MAX_P];
	void solve();
}

/*
开关问题 - POJ 3276. Face the Right Way
*/
namespace p150 {
	const int MAX_N = 5000;
	extern int N;
	extern int dir[MAX_N];

	void solve();
}

/*
开关问题 - POJ 3279. Fliptile
*/
namespace p153 {
	const int MAX_M = 15, MAX_N = 15;
	extern int M, N;
	extern int tile[MAX_M][MAX_N];

	void solve();
}

/*
弹性碰撞 - POJ 3684. Physics Experiment
*/
namespace p158 {
	const int MAX_N = 100;
	extern int N,H,R,T;
	void solve();
}

/*
折半枚举 - POJ 2785. 4 values whose sum is 0
*/
namespace p160 {
	const int  MAX_N = 4000;

	extern int n;
	extern int A[MAX_N], B[MAX_N], C[MAX_N], D[MAX_N];
	void solve();
}

/*
折半枚举 - 超大背包问题
*/
namespace p162 {
	const int  INF = 1000000;
	const int  MAX_N = 40;

	extern int n;
	extern LL w[MAX_N], v[MAX_N], W;
	void solve();
}

/*
坐标离散化 - 区域的个数
*/
namespace p164 {
	const int MAX_N = 500;

	extern int W, H, N;
	extern int X1[MAX_N], X2[MAX_N], Y1[MAX_N], Y2[MAX_N];
	void solve();
}

/*
线段树 - RMQ问题
*/
namespace p169 {
	const int MAX_N = 1 << 17; // 取一个大于N的2的幂

	void init(int n_);
	void update(int k, int a);
	int query(int a, int b, int k, int l, int r);
}

/*
线段树 - POJ 2991. Crane
*/
namespace p170 {
	const int MAX_N = 10000;
	const int MAX_C = 10000;

	extern int N, C;
	extern int L[MAX_N];
	extern int S[MAX_C], A[MAX_N];

	void solve();
}

/*
树状数组
*/
namespace p177 {
	struct BIT {
		VI a;
		BIT(int n) { a = VI(n + 1); }
		int query(int i) { int s = 0; while (i > 0) s += a[i], i -= (i & -i); return s; }
		void add(int i, int v) { while (i < a.size()) a[i] += v, i += (i & -i); }
	};
}

/*
线段树 - POj 3468. 一个简单的整数问题
*/
namespace p179 {
	
}

/*
树状数组 - POj 3468. 一个简单的整数问题
*/
namespace p181 {

}

/*
平方分割 - POj 2104. K-th number
*/
namespace p185 {

	const int MAX_N = 100000;
	const int MAX_M = 5000;

	extern int N, M;
	extern int A[MAX_N];
	extern int I[MAX_M], J[MAX_M], K[MAX_M];

	void solve();
}

/*
线段树 - POj 2104. K-th number
*/
namespace p188 {
	const int MAX_N = 100000;
	const int MAX_M = 5000;

	extern int N, M;
	extern int A[MAX_N];
	extern int I[MAX_M], J[MAX_M], K[MAX_M];

	void solve();
}

/*
状压DP - 旅行商问题
*/
namespace p191 {
	const int MAX_N = 15;
	const int INF = 1e9;

	extern int n;
	extern int d[MAX_N][MAX_N];
	void solve();
}

/*
状压DP - POJ 2686. Travelling by Stagecoach
*/
namespace p193 {

	const int MAX_N = 8;
	const int MAX_M = 30;
	const int INF = 1e9;

	extern int n, m, a, b;
	extern int t[MAX_N];
	extern int d[MAX_M][MAX_M];
	void solve();
}

/*
状压DP - 铺砖问题
*/
namespace p196 {

	const int MAX_N = 15;
	const int MAX_M = 15;
	const int INF = 1e9;

	extern int n, m, M;
	extern int color[MAX_N][MAX_M];
	void solve();
}

/*
矩阵幂 - 斐波那契数列
*/
namespace p199_part1 {
	typedef vector<int> vec;
	typedef vector<vec> mat;
	mat operator*(const mat& a, const mat& b);
	mat pow(mat a, LL n);
}


/*
矩阵幂 - 斐波那契数列
*/
namespace p199_part2 {
	extern LL n;
	void solve();
}

/*
矩阵幂 - POJ 3734. Blocks
*/
namespace p202 {
	extern int N;
	void solve();
}

/*
矩阵幂 - 长度为K的路径计数
*/
namespace p203 {

	const int MAX_N = 100;

	extern int n, k;
	extern int g[MAX_N][MAX_N];
	void solve();
}

/*
矩阵幂 - POJ 3233. Matrix Power Series
*/
namespace p204 {

	using namespace p199_part1;

	extern int n, k, M;
	extern mat A;
	void solve();
}

/*
最大流 - Ford-Fulkerson算法
*/
namespace p209 {

	const int INF = 1e9;
	const int MAX_V = 1;

	struct edge { int to, cap, rev; };

	extern vector<edge> G[MAX_V];
	extern int max_flow(int s, int t);
}

/*
最大流 - Dinic算法
*/
namespace p216 {

	const int INF = 1e9;
	const int MAX_V = 1;

	struct edge { int to, cap, rev; };

	extern vector<edge> G[MAX_V];
	extern int max_flow(int s, int t);
}

/*
二分图匹配 - 指派问题
*/
namespace p219 {

	const int MAX_V = 1;

	extern int V;
	extern vector<int> G[MAX_V];

	void add_edge(int u, int v);
	int biparite_matching();
}

/*
最小费用流
*/
namespace p222 {

	const int INF = 1e9;
	const int MAX_V = 1;

	struct edge { int to, cap, cost, rev; };
	extern int V;
	extern vector<edge> G[MAX_V];

	int min_cost_flow(int s, int t, int f);
}

/*
二分图匹配 - POJ 3041. Asteroids
*/
namespace p228 {
	const int MAX_K = 10000;
	extern int N, K, R[MAX_K], C[MAX_K];

	void solve();
}

/*
二分图匹配 - POJ 3057. Evacuation
*/
namespace p230 {
	const int MAX_X = 12, MAX_Y = 12;
	extern int X, Y;
	extern char field[MAX_X][MAX_Y + 1];

	void solve();
}

/*
最大流 - POJ 3281. Dining
*/
namespace p234 {
	const int MAX_N = 100, MAX_F = 100, MAX_D = 100;

	extern int N, F, D;
	extern bool likeF[MAX_N][MAX_F];
	extern bool likeD[MAX_N][MAX_D];

	void solve();
}

/*
最大流 - POJ 3469. Dual Core CPU
*/
namespace p236 {
	const int MAX_N = 20000;
	const int MAX_M = 200000;

	extern int N, M;
	extern int A[MAX_N], B[MAX_N];
	extern int a[MAX_M], b[MAX_M], w[MAX_M];

	void solve();
}

/*
最小费用流 - POJ 2135. Farm Tour
*/
namespace p238 {
	const int MAX_M = 10000;

	extern int N, M;
	extern int a[MAX_M], b[MAX_M], c[MAX_M];

	void solve();
}

/*
最小费用流 - POJ 2135. Evacuation Plan,方法一
*/
namespace p240 {
	const int MAX_N = 100, MAX_M = 100;

	extern int N, M;
	extern int X[MAX_N], Y[MAX_N], B[MAX_N];
	extern int P[MAX_M], Q[MAX_M], C[MAX_M];
	extern int E[MAX_N][MAX_N];

	void solve();
}

/*
最小费用流 - POJ 2135. Evacuation Plan,方法二
*/
namespace p242 {
	const int MAX_N = 100, MAX_M =100;

	extern int N, M;
	extern int X[MAX_N], Y[MAX_N], B[MAX_N];
	extern int P[MAX_M], Q[MAX_M], C[MAX_M];
	extern int E[MAX_N][MAX_N];

	void solve();
}

/*
最小费用流 - POJ 3686. The Windy's
*/
namespace p243 {
	const int MAX_N = 50, MAX_M = 50;

	extern int N, M;
	extern int Z[MAX_N][MAX_M];

	void solve();
}

/*
最小费用流 - POJ 3680. Intervals
*/
namespace p246 {
	const int MAX_N = 200;

	extern int N, K;
	extern int a[MAX_N], b[MAX_N], w[MAX_N];

	void solve();
}

/*
计算几何基础 - 点与线段
*/
namespace p250_part1 {
	struct P {
		double x, y;
		P();
		P(double x, double y);
		P operator+(P p);
		P operator-(P p);
		P operator*(double d);
		double dot(P p);
		double det(P p);
	};
	bool on_seg(P p1, P p2, P q);// 判断点q是否在直线上
	P intersection(P p1, P p2, P q1, P q2);// 计算两直线的交点
}

/*
计算几何基础 - POJ 1127. Jack Straws
*/
namespace p250_part2 {
	const int MAX_N = 12, MAX_M = 10000;
	
	using namespace p250_part1;

	extern int n;
	extern P p[MAX_N], q[MAX_N];
	extern int m;
	extern int a[MAX_M], b[MAX_M];

	void solve();
}

/*
极限情况 - AOJ 2308. White Bird
*/
namespace p255 {

}

/*
平面扫描 - POJ 2932. Coneology
*/
namespace p258 {
	const int MAX_N = 40000;
	extern int N;
	extern double x[MAX_N], y[MAX_N], r[MAX_N];
	void solve();
}

/*
凸包 - POJ 2187. Beauty Contest
*/
namespace p260 {
	using namespace p250_part1;

	const int MAX_N = 50000;
	extern int N;
	extern P ps[MAX_N];
	void solve();
}

/*
3.6.5数值积分 - AOJ 1313. Intersection of Two Prisms
*/
namespace p263 {
	const int MAX_M = 100;
	const int MAX_N = 100;
	const int INF = 1e9;

	extern int M, N;
	extern int X1[MAX_M], Y1[MAX_M];
	extern int X2[MAX_N], Z2[MAX_N];

	void solve();
}

/*
高斯消元法
*/
namespace p286 {
	typedef vector<double> vec;
	typedef vector<vec> mat;

	vec gauss_jordan(const mat& A, const vec& b);
}

/*
高斯消元法 - Random Walk
*/
namespace p288 {
	
}

/*
模运算的世界
*/
namespace p291 {
	int mod_inverse(int a, int m);
}
