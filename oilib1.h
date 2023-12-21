#pragma once

// ��ʽͼ��dfs - ���ֺ�����
namespace p30 {
	const int MAX_N = 20;
	
	extern int a[MAX_N];
	extern int n, k;

	void solve(); 
}

// �˷���dfs - Lake Counting (POJ 2386)
namespace p32 {
	const int MAX_N = 20;
	const int MAX_M = 20;
	
	extern char field[MAX_N][MAX_M];
	extern int n, m;

	void solve();
}

// bfs - �Թ����·��
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

// DP - 01����
namespace p51 {

	const int MAX_N = 100;
	const int MAX_W = 1000;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - �����������
namespace p56 {

	const int maxn = 1000;
	const int maxm = 1000;

	extern int n, m;
	extern string s, t;

	void solve();
}

// DP - ��ȫ����
namespace p57 {

	const int MAX_N = 100;
	const int MAX_W = 1000;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - 01��������
namespace p60 {

	const int MAX_N = 100;

	extern int n, W;
	extern int w[MAX_N], v[MAX_N];

	void solve();
}

// DP - ���ز��ֺ�����
namespace p62 {
	const int MAX_N = 100;
	
	extern int n, K;
	extern int a[MAX_N], m[MAX_N];

	void solve();
}

// DP - ���ز��ֺ������Ż�
namespace p63 {
	const int MAX_N = 100;

	extern int n, K;
	extern int a[MAX_N], m[MAX_N];

	void solve();
}

// DP - ���������������
namespace p64 {
	const int MAX_N = 1000;

	extern int n;
	extern int a[MAX_N];

	void solve();
}

// DP - ����������������Ż�
namespace p65 {
	const int MAX_N = 1000;

	extern int n;
	extern int a[MAX_N];

	void solve();
}

// DP - ������
namespace p66 {

	const int MAX_N = 1000;
	const int MAX_M = 1000;

	extern int n, m, M;

	void solve();
}

// DP - ���ؼ������
namespace p68 {

	const int MAX_N = 1000;
	const int MAX_M = 1000;

	extern int n, m, a[MAX_N], M;

	void solve();
}

/*
���鼯
*/
namespace p84 {

	const int MAX_N = 1;

	extern int par[MAX_N], rank[MAX_N];
	void init(int n);
	int find(int u);
	void unite(int u, int v);
	bool same(int u, int v);
}


// ���鼯 - POJ 1182. ʳ����
namespace p88 {

	const int MAX_K = 100000;

	extern int N, K;
	extern int T[MAX_K], X[MAX_K], Y[MAX_K];//��i����Ϣ������,X_i,y_i

	void solve();
}

// ͼ�㷨 - ����ͼ�ж�
namespace p97 {
	const int MAX_V = 1000;

	extern VI G[MAX_V];	//ͼ
	extern int V;		//������
	
	void solve();
}

// ͼ�㷨 - Bellmsn-Ford�㷨
namespace p100 {
	const int INF = 1000000;
	
	struct edge { int from, to, cost; };
	extern vector<edge> es;	//ͼ
	extern int V,E;			//������

	void shortest_path(int s);
	bool find_negative_loop();
}

// ͼ�㷨 - �ڽӾ���Dijkstra�㷨
namespace p101 {
	const int INF = 1000000;
	const int MAX_V = 100;

	extern int cost[MAX_V][MAX_V];	//ͼ
	extern int V;					//������

	void dijkstra(int s);
}

// ͼ�㷨 - �ڽӱ�Dijkstra�㷨
namespace p102 {
	const int INF = 1000000;
	const int MAX_V = 1000;

	struct edge { int to, cost; };
	extern vector<edge> G[MAX_V];	//ͼ
	extern int V;			//������

	void dijkstra(int s);
}

// ͼ�㷨 - Floyd�㷨
namespace p103 {
	const int INF = 1000000;
	const int MAX_V = 1000;

	extern int d[MAX_V][MAX_V];	//ͼ
	extern int V;			//������

	void floyd(int s);
}

// ͼ�㷨 - ·����ԭ
namespace p104 {
	const int INF = 1000000;
	const int MAXV = 100;

	extern int cost[MAXV][MAXV];	//ͼ
	extern int V;					//������

	void dijkstra(int s);
	vector<int> get_path(int t);
}

// ͼ�㷨 - ��С������Prim�㷨
namespace p105 {
	const int INF = 1000000;
	const int MAXV = 100;

	extern int cost[MAXV][MAXV];	//ͼ
	extern int V;					//������

	void prim(int s);
}

// ͼ�㷨 - ��С������Kruskal�㷨
namespace p107 {

	struct edge { int u, v, cost; };
	extern vector<edge> es;		//ͼ
	extern int V, E;			//������,����

	int kruskal(int s);
}

// ŷ������㷨
namespace p113 {
	int gcd(int a, int b); // ��(a,b)
}

// ��չŷ������㷨
namespace p115 {
	int extgcd(int a, int b, int& x, int& y); // ��x,yʹ��ax+by=1
}

// �����ж�
namespace p117 {
	bool is_prime(int n); // �����ж�
	vector<int> divisor(int n); // Լ��ö��
	map<int, int> prime_factor(int n); // �������ֽ�
}

// ��ѧ�㷨 - ������ɫ��ɸ��
namespace p118 {

	const int MAX_N = 1000000;
	extern int prime[MAX_N];
	extern int is_prime[MAX_N+1];

	int sieve(int n);
}

// ��ѧ�㷨 - ����ɸ��
namespace p120 {

	const int MAX_L = 1000000;
	const int MAX_SQRT_B = 1000000;

	extern bool is_prime[MAX_L];
	extern bool is_prime_small[MAX_SQRT_B + 1];

	void segment_sieve(LL a, LL b);
}

// ��ѧ�㷨 - ������
namespace p122 {

	LL mod_pow(LL x, LL n, LL m);
}

// lower_bound��ʵ��
namespace p138 {
	const int MAX_N = 1000000;
	extern int n, k, a[MAX_N];

	void solve();
}

// ���ִ� - Cable master (POJ 1064)
namespace p140 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int N, K;
	extern double L[MAX_N];

	void solve();
}

// �����Сֵ - Aggressive Cows (POJ 2456)
namespace p142 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int N, M, x[MAX_N];

	void solve();
}

// ���ƽ��ֵ
namespace p143 {
	const int INF = 1000000;
	const int MAX_N = 10000;
	extern int n, k, w[MAX_N], v[MAX_N];

	void solve();
}

/*
��ȡ�� -  POJ 3061. Subsequence
��������Ϊn������a������S���ҳ���̵��ܺͲ�С��S�����������С�
*/
namespace p146 {

	const int MAX_N = 100000;

	extern int n, S, a[MAX_N];

	void solve();
}

/*
��ȡ�� -  POJ 3320. Jessica's Reading Problem
*/
namespace p149 {

	const int MAX_P = 1000000;
	extern int P;
	extern int a[MAX_P];
	void solve();
}

/*
�������� - POJ 3276. Face the Right Way
*/
namespace p150 {
	const int MAX_N = 5000;
	extern int N;
	extern int dir[MAX_N];

	void solve();
}

/*
�������� - POJ 3279. Fliptile
*/
namespace p153 {
	const int MAX_M = 15, MAX_N = 15;
	extern int M, N;
	extern int tile[MAX_M][MAX_N];

	void solve();
}

/*
������ײ - POJ 3684. Physics Experiment
*/
namespace p158 {
	const int MAX_N = 100;
	extern int N,H,R,T;
	void solve();
}

/*
�۰�ö�� - POJ 2785. 4 values whose sum is 0
*/
namespace p160 {
	const int  MAX_N = 4000;

	extern int n;
	extern int A[MAX_N], B[MAX_N], C[MAX_N], D[MAX_N];
	void solve();
}

/*
�۰�ö�� - ���󱳰�����
*/
namespace p162 {
	const int  INF = 1000000;
	const int  MAX_N = 40;

	extern int n;
	extern LL w[MAX_N], v[MAX_N], W;
	void solve();
}

/*
������ɢ�� - ����ĸ���
*/
namespace p164 {
	const int MAX_N = 500;

	extern int W, H, N;
	extern int X1[MAX_N], X2[MAX_N], Y1[MAX_N], Y2[MAX_N];
	void solve();
}

/*
�߶��� - RMQ����
*/
namespace p169 {
	const int MAX_N = 1 << 17; // ȡһ������N��2����

	void init(int n_);
	void update(int k, int a);
	int query(int a, int b, int k, int l, int r);
}

/*
�߶��� - POJ 2991. Crane
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
��״����
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
�߶��� - POj 3468. һ���򵥵���������
*/
namespace p179 {
	
}

/*
��״���� - POj 3468. һ���򵥵���������
*/
namespace p181 {

}

/*
ƽ���ָ� - POj 2104. K-th number
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
�߶��� - POj 2104. K-th number
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
״ѹDP - ����������
*/
namespace p191 {
	const int MAX_N = 15;
	const int INF = 1e9;

	extern int n;
	extern int d[MAX_N][MAX_N];
	void solve();
}

/*
״ѹDP - POJ 2686. Travelling by Stagecoach
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
״ѹDP - ��ש����
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
������ - 쳲���������
*/
namespace p199_part1 {
	typedef vector<int> vec;
	typedef vector<vec> mat;
	mat operator*(const mat& a, const mat& b);
	mat pow(mat a, LL n);
}


/*
������ - 쳲���������
*/
namespace p199_part2 {
	extern LL n;
	void solve();
}

/*
������ - POJ 3734. Blocks
*/
namespace p202 {
	extern int N;
	void solve();
}

/*
������ - ����ΪK��·������
*/
namespace p203 {

	const int MAX_N = 100;

	extern int n, k;
	extern int g[MAX_N][MAX_N];
	void solve();
}

/*
������ - POJ 3233. Matrix Power Series
*/
namespace p204 {

	using namespace p199_part1;

	extern int n, k, M;
	extern mat A;
	void solve();
}

/*
����� - Ford-Fulkerson�㷨
*/
namespace p209 {

	const int INF = 1e9;
	const int MAX_V = 1;

	struct edge { int to, cap, rev; };

	extern vector<edge> G[MAX_V];
	extern int max_flow(int s, int t);
}

/*
����� - Dinic�㷨
*/
namespace p216 {

	const int INF = 1e9;
	const int MAX_V = 1;

	struct edge { int to, cap, rev; };

	extern vector<edge> G[MAX_V];
	extern int max_flow(int s, int t);
}

/*
����ͼƥ�� - ָ������
*/
namespace p219 {

	const int MAX_V = 1;

	extern int V;
	extern vector<int> G[MAX_V];

	void add_edge(int u, int v);
	int biparite_matching();
}

/*
��С������
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
����ͼƥ�� - POJ 3041. Asteroids
*/
namespace p228 {
	const int MAX_K = 10000;
	extern int N, K, R[MAX_K], C[MAX_K];

	void solve();
}

/*
����ͼƥ�� - POJ 3057. Evacuation
*/
namespace p230 {
	const int MAX_X = 12, MAX_Y = 12;
	extern int X, Y;
	extern char field[MAX_X][MAX_Y + 1];

	void solve();
}

/*
����� - POJ 3281. Dining
*/
namespace p234 {
	const int MAX_N = 100, MAX_F = 100, MAX_D = 100;

	extern int N, F, D;
	extern bool likeF[MAX_N][MAX_F];
	extern bool likeD[MAX_N][MAX_D];

	void solve();
}

/*
����� - POJ 3469. Dual Core CPU
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
��С������ - POJ 2135. Farm Tour
*/
namespace p238 {
	const int MAX_M = 10000;

	extern int N, M;
	extern int a[MAX_M], b[MAX_M], c[MAX_M];

	void solve();
}

/*
��С������ - POJ 2135. Evacuation Plan,����һ
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
��С������ - POJ 2135. Evacuation Plan,������
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
��С������ - POJ 3686. The Windy's
*/
namespace p243 {
	const int MAX_N = 50, MAX_M = 50;

	extern int N, M;
	extern int Z[MAX_N][MAX_M];

	void solve();
}

/*
��С������ - POJ 3680. Intervals
*/
namespace p246 {
	const int MAX_N = 200;

	extern int N, K;
	extern int a[MAX_N], b[MAX_N], w[MAX_N];

	void solve();
}

/*
���㼸�λ��� - �����߶�
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
	bool on_seg(P p1, P p2, P q);// �жϵ�q�Ƿ���ֱ����
	P intersection(P p1, P p2, P q1, P q2);// ������ֱ�ߵĽ���
}

/*
���㼸�λ��� - POJ 1127. Jack Straws
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
������� - AOJ 2308. White Bird
*/
namespace p255 {

}

/*
ƽ��ɨ�� - POJ 2932. Coneology
*/
namespace p258 {
	const int MAX_N = 40000;
	extern int N;
	extern double x[MAX_N], y[MAX_N], r[MAX_N];
	void solve();
}

/*
͹�� - POJ 2187. Beauty Contest
*/
namespace p260 {
	using namespace p250_part1;

	const int MAX_N = 50000;
	extern int N;
	extern P ps[MAX_N];
	void solve();
}

/*
3.6.5��ֵ���� - AOJ 1313. Intersection of Two Prisms
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
��˹��Ԫ��
*/
namespace p286 {
	typedef vector<double> vec;
	typedef vector<vec> mat;

	vec gauss_jordan(const mat& A, const vec& b);
}

/*
��˹��Ԫ�� - Random Walk
*/
namespace p288 {
	
}

/*
ģ���������
*/
namespace p291 {
	int mod_inverse(int a, int m);
}
