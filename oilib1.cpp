#include "types.h"
#include "oilib1.h"

namespace lib1 {

namespace p30 {
	int a[MAX_N];
	int n, k;

	bool dfs(int i, int sum) {
		if (i == n) return sum == k;
		if (dfs(i + 1, sum)) return true;
		if (dfs(i + 1, sum + a[i])) return true;
		return false;
	}

	void solve() {
		if (dfs(0, 0)) cout << "Yes" << endl;
		else cout << "No" << endl;
	}
}

namespace p32 {
	int dir[8][2] =
	{
		{-1,-1},{0,-1},{1,-1},
		{-1,0},{1,0},
		{-1,1},{0,1},{1,1},
	};

	char field[MAX_N][MAX_M];
	int n, m;

	int dfs(int x, int y)
	{
		int nx, ny;
		field[x][y] = '.';
		for (int d = 0; d < 8; d++)
		{
			nx = x + dir[d][0];
			ny = y + dir[d][1];
			if (nx >= 1 && nx <= n && ny >= 1 && ny <= m && field[nx][ny] == 'W')
				dfs(nx, ny);
		}
		return 0;
	}

	void solve() {
		int ans = 0;
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= m; j++)
			{
				if (field[i][j] == 'W')
				{
					dfs(i, j);
					ans++;
				}
			}
		}
		cout << ans << endl;
	}
}

namespace p34 {

	char maze[MAXN][MAXM+1];
	int N, M;
	int sx, sy;
	int gx, gy;
	int d[MAXN][MAXM];
	int dx[4] = { 1,0,-1,0 }, dy[4] = { 0,1,0,-1 };

	int bfs()
	{
		queue<PII> que;
		rep(i, 0, N)
			rep(j, 0, M)
				d[i][j] = INF;
		que.push({ sx,sy });
		d[sx][sy] = 0;

		while (que.size()) {
			PII p = que.front(); que.pop();
			if (p.first == gx && p.second == gy) break;
			rep(i, 0, 4) {
				int nx = p.first + dx[i], ny = p.second + dy[i];
				if (nx >= 0 && nx < N && ny >= 0 && ny < M && maze[nx][ny] != '#' && d[nx][ny] == INF) {
					que.push({ nx,ny });
					d[nx][ny] = d[p.first][p.second] + 1;
				}
			}
		}
		return d[gx][gy];
	}

	void solve() {
		int res = bfs();
		cout << res << endl;
	}
}

namespace p51 {

	int n, W;
	int w[MAX_N], v[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void solve() {
		repd(i, n - 1, -1)
			rep(j, 0, W + 1)
			if (j < w[i])
				dp[i][j] = dp[i + 1][j];
			else
				dp[i][j] = max(dp[i + 1][j], dp[i + 1][j - w[i]] + v[i]);
		cout << dp[n][W];
	}
}

namespace p56 {

	int n, m;
	string s, t;
	int dp[maxn + 1][maxm + 1];

	void solve() {
		rep(i, 0, n)
			rep(j, 0, m)
			if (s[i] == t[j])
				dp[i + 1][j + 1] = dp[i][j] + 1;
			else
				dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
		cout << dp[n][m];
	}
}

namespace p57 {

	int n, W;
	int w[MAX_N], v[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void solve() {
		rep(i, 0, n)
			rep(j, 0, W + 1)
			if (j < w[i])
				dp[i + 1][j] = dp[i][j];
			else
				dp[i + 1][j] = max(dp[i][j], dp[i + 1][j - w[i]] + v[i]);
		cout << dp[n][W];
	}
}

namespace p60 {

	const int MAX_V = 100;
	const int INF = 1000000;

	int n, W;
	int w[MAX_N], v[MAX_N];
	int dp[MAX_N + 1][MAX_N * MAX_V + 1];

	void solve() {
		fill(dp[0], dp[0] + MAX_N * MAX_V + 1, INF);
		dp[0][0] = 0;
		rep(i, 0, n)
			rep(j, 0, MAX_N * MAX_V + 1)
				if (j < v[i])
					dp[i + 1][j] = dp[i][j];
				else
					dp[i + 1][j] = min(dp[i][j], dp[i][j - v[i]] + w[i]);
		int res = 0;
		rep(i, 0, MAX_N * MAX_V+1)
			if (dp[n][i] <= W)
				res = i;
		cout << res << endl;
	}
}

namespace p62 {

	const int MAX_K = 100000;

	int n, K;
	int a[MAX_N], m[MAX_N];
	bool dp[MAX_N + 1][MAX_K + 1];

	void solve() {
		dp[0][0] = true;
		rep(i, 0, n)
			rep(j, 0, K + 1) 
				for (int k = 0; k <= m[i] && k * a[i] <= j; k++)
					dp[i + 1][j] |= dp[i][j - k * a[i]];

		if (dp[n][K])
			cout << "Yes" << endl;
		else
			cout << "No" << endl;
	}
}

namespace p63 {

	const int MAX_K = 100000;

	int n, K;
	int a[MAX_N], m[MAX_N];
	int dp[MAX_K + 1];

	void solve() {
		memset(dp, -1, sizeof(dp));
		dp[0] = 0;
		rep(i, 0, n)
			rep(j, 0, K + 1)
				if (dp[j] >= 0)
					dp[j] = m[i];
				else if (j < a[i] || dp[j - a[i]] <= 0)
					dp[j] = -1;
				else
					dp[j] = dp[j - a[i]] - 1;

		if (dp[K] >= 0)
			cout << "Yes" << endl;
		else
			cout << "No" << endl;
	}
}

namespace p64 {
	
	int n;
	int a[MAX_N];
	int dp[MAX_N];

	void solve() {
		int res = 0;
		rep(i, 0, n) {
			dp[i] = 1;
			rep(j, 0, i+1) {
				if (a[j] < a[i])
					dp[i] = max(dp[i], dp[j] + 1);
			}
			res = max(res, dp[i + 1]);
		}
		cout << res << endl;
	}
}

namespace p65 {

	const int INF = 1000000;

	int n;
	int a[MAX_N];
	int dp[MAX_N];

	void solve() {
		fill(dp, dp + n, INF);
		dp[0] = 0;
		rep(i, 0, n) {
			*lower_bound(dp, dp + n, a[i]) = a[i];
		}
		cout << lower_bound(dp, dp + n, INF) - dp << endl;
	}
}

namespace p66 {

	int n, m, M;
	int dp[MAX_M + 1][MAX_N + 1];

	void solve() {
		dp[0][0] = 1;

		rep(i, 1, m + 1)
			rep(j, 0, n + 1)
				if (j - i >= 0)
					dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % M;
				else
					dp[i][j] = dp[i - 1][j];

		cout << dp[m][n] << endl;
	}
}

namespace p68 {

	int n, m, a[MAX_N], M;
	int dp[MAX_N + 1][MAX_M + 1];

	void solve() {
		dp[0][0] = 1;
		rep(i, 0, n) {
			dp[i][0] = 1;
			rep(j, 1, m + 1) {
				if (j - 1 - a[i] >= 0)
					dp[i + 1][j] = (dp[i + 1][j - 1] + dp[i][j] - dp[i][j - 1 - a[i]] + M) % M;
				else
					dp[i + 1][j] = (dp[i + 1][j - 1] + dp[i][j]) % M;
			}
		}
		cout << dp[n][m] << endl;
	}
}

namespace p87 {

	int par[MAX_N], rank[MAX_N];

	void init(int n) { 
		rep(i, 0, n) {
			par[i] = i;
			rank[i] = 0;
		}
	}
	int find(int u) { 
		return (par[u] == u) ? u : par[u] = find(par[u]); 
	}
	
	void unite(int u, int v) {
		u = find(u);
		v = find(v);
		if (u == v) return;
		if (rank[u] < rank[v])
			par[u] = v;
		else {
			par[v] = u;
			if (rank[u] == rank[v]) rank[u]++;
		}
	}

	bool same(int u, int v) { 
		return find(u) == find(v); 
	}
}

namespace p88 {

	using namespace p87;

	int N, K;
	int T[MAX_K], X[MAX_K], Y[MAX_K];//第i条信息的类型,X_i,y_i

	void solve() {
		// x,x+N,x+2*N分别表示x-A,x-B,x-C
		init(3 * N);

		int ans = 0;
		rep(i, 0, K) {
			int t = T[i], x = X[i]-1, y = Y[i]-1;

			if (x < 0 || x >= N || y < 0 || y >= N) {
				ans++;
				continue;
			}

			if (t == 1) { // x和y同类
				if (same(x,y+N) || same(x,y+2*N)) {
					ans++;
					continue;
				}
				unite(x,y);
				unite(x+N,y+N);
				unite(x+2*N,y+2*N);
			}
			else { // x吃y
				if (same(x, y) || same(x, y+2*N)) {
					ans++;
					continue;
				}
				unite(x, y+N);
				unite(x + N, y + 2*N);
				unite(x + 2 * N, y);
			}
		}

		cout << ans << endl;
	}
}

namespace p97 {
	VI G[MAX_V];
	int V;
	int color[MAX_V];		//顶点i的颜色(1或-1)

	bool dfs(int v, int c) {
		color[v] = c;
		for (int i = 0; i < G[v].size(); i++) {
			if (color[G[v][i]] == c) return false;
			if (color[G[v][i]] == 0 && !dfs(G[v][i], -c)) return false;
		}
		return true;
	}

	void solve() {
		rep(i, 0, V) {
			if (color[i] == 0) {
				if (!dfs(i, 1)) {
					cout << "No" << endl;
					return;
				}
			}
		}
		cout << "Yes" << endl;
	}
}

namespace p100 {

	vector<edge> es;	//图
	int V, E;			//顶点数
	VI d;

	void shortest_path(int s) {
		d = VI(V);
		rep(i, 0, V) d[i] = INF;
		d[s] = 0;
		for (;;) {
			bool update = false;
			rep(i, 0, E) {
				edge e = es[i];
				if (d[e.from] != INF && d[e.to] > d[e.from] + e.cost) {
					d[e.to] = d[e.from] + e.cost;
					update = true;
				}
			}
			if (!update) break;
		}
	}
	bool find_negative_loop() {
		rep(i, 0, V) d[i] = 0;
		rep(i, 0, V)
			rep(j, 0, E) {
			edge e = es[j];
			if (d[e.to] > d[e.from] + e.cost) {
				d[e.to] = d[e.from] + e.cost;
				if (i == V - 1) return true;
			}
		}
		return false;
	}
}

namespace p101 {

	int cost[MAX_V][MAX_V];	//图
	int V;					//顶点数

	void dijkstra(int s) {

	}
}

namespace p102 {
	
	vector<edge> G[MAX_V];	//图
	int V;			//顶点数
	int d[MAX_V];

	void dijkstra(int s) {
		priority_queue<PII, vector<PII>, greater<PII> > que;
		rep(i, 0, V)d[i] = INF;
		d[s] = 0;
		que.push(PII(0, s));
		while (!que.empty()) {
			PII p = que.top(); que.pop();
			int v = p.second;
			if (d[v] < p.first) continue;
			rep(i, 0, G[v].size()) {
				edge e = G[v][i];
				if (d[e.to] > d[v] + e.cost) {
					d[e.to] = d[v] + e.cost;
					que.push(PII(d[e.to], e.to));
				}
			}
		}
	}
}

namespace p103 {

	int d[MAX_V][MAX_V];	//图
	int V;			//顶点数

	void floyd(int s) {
		rep(k, 0, V) 
			rep(i, 0, V) 
				rep(j, 0, V) 
					d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
	}
}

namespace p104 {

	int cost[MAX_V][MAX_V];
	int d[MAX_V];
	bool used[MAX_V];
	int V;

	int pre[MAX_V];

	void dijkstra(int s)
	{
		fill(d, d + V, INF);
		fill(used, used + V, false);
		d[s] = 0;
		for (;;) {
			int v = -1;
			rep(u, 0, V)
				if (!used[u] && (v == -1 || d[u] < d[v])) v = u;
			if (v == -1) break;
			used[v] = true;
			rep(u, 0, V) {
				d[u] = min(d[u], d[v] + cost[v][u]);
				pre[u] = v;
			}
		}
	}

	VI get_path(int t) {
		vector<int> path;
		for (; t != -1; t = pre[t])
			path.push_back(t);
		reverse(path.begin(), path.end());
		return path;
	}
}

namespace p105 {

	int cost[MAX_V][MAX_V], V;
	int mincost[MAX_V];
	bool used[MAX_V];

	int prim(int s) {
		rep(i, 0, V) {
			mincost[i] = INF;
			used[i] = false;
		}
		mincost[0] = 0;
		int res = 0;
		while (true) {
			int v = -1;
			rep(u, 0, V)
				if (!used[u] && (v == -1 || mincost[u] < mincost[v])) v = u;
			if (v == -1)break;
			used[v] = true;
			res += mincost[v];
			rep(u, 0, V)
				mincost[u] = min(mincost[u], cost[v][u]);
		}
		return res;
	}
}

namespace p107 {

	using namespace p87;

	vector<edge> es;		//图
	int V, E;			//顶点数,边数

	bool comp(const edge& e1, const edge& e2) { return e1.cost < e2.cost; }

	int kruskal(int s) {
		sort(es.begin(), es.begin() + E, comp);
		init(V);
		int res = 0;
		rep(i, 0, E) {
			edge e = es[i];
			if (!same(e.u, e.v)) {
				unite(e.u, e.v);
				res += e.cost;
			}
		}
		return res;
	}
}

namespace p114 {
	int gcd(int a, int b) {
		if (b == 0) return a;
		return gcd(b, a % b);
	}
}

namespace p115 {
	int extgcd(int a, int b, int& x, int& y) {
		int d = a;
		if (b != 0) {
			d = extgcd(b, a % b, y, x);
			y -= (a / b) * x;
		}
		else {
			x = 1; y = 0;
		}
		return d;
	}
}

namespace p117 {
	bool is_prime(int n) {
		for (int i = 2; i * i <= n; i++)
			if (n % i == 0) return false;
		return true;
	}

	// 约数枚举
	vector<int> divisor(int n) {
		vector<int> res;
		for (int i = 1; i * i <= n; i++)
			if (n % i == 0) {
				res.push_back(i);
				if (i != n / i) res.push_back(n / i);
			}
		return res;
	}

	// 质因数分解
	map<int, int> prime_factor(int n) {
		map<int, int> res;
		for (int i = 2; i * i <= n; i++)
			while (n % i == 0) {
				++res[i];
				n /= i;
			}
		if (n != 1) res[n] = 1;
		return res;
	}
}

namespace p118 {

	int prime[MAX_N];
	int is_prime[MAX_N + 1];

	int sieve(int n) {
		int p = 0;
		fill(is_prime, is_prime + n, true);
		is_prime[0] = is_prime[1] = false;
		rep(i, 2, n + 1) {
			if (is_prime[i]) {
				prime[p++] = i;
				for (int j = 2 * i; j <= n; j += i)
					is_prime[j] = false;
			}
		}
		return p;
	}

}

namespace p120 {

	bool is_prime[MAX_L];
	bool is_prime_small[MAX_SQRT_B + 1];

	void segment_sieve(LL a, LL b) {
		for (int i = 0; (LL)i * i < b; i++) is_prime_small[i] = true;
		for (int i = 0; i < b - a; i++) is_prime[i] = true;
		for (int i = 2; (LL)i * i < b; i++) {
			if (is_prime_small[i]) {
				for (int j = 2 * i; (LL)j * j < b; j += i) is_prime_small[j] = false;
				for (LL j = max(2LL, (a + i - 1) / i) * i; j < b; j += i) is_prime[j - a] = false;
			}
		}
	}
}

namespace p122 {

	LL mod_pow(LL x, LL n, LL m)
	{
		LL res = 1;
		while (n > 0)
		{
			if (n & 1) res = res * x % m;
			x = x * x % m;
			n >>= 1;
		}
		return res;
	}
}

namespace p138 {
	int n, k, a[MAX_N];

	void solve()
	{
		int lb = -1, ub = n - 1;
		while (ub - lb > 1) {
			int mid = (ub + lb) / 2;
			if (a[mid] >= k) // 由此可见,lower_bound实质是最小化答案,这里的答案指的是k的索引
				ub = mid;
			else
				lb = mid;
		}
		cout << ub << endl;
	}

}

namespace p140 {
	
	int N, K;
	double L[MAX_N];

	bool C(double x) {
		int num = 0;
		rep(i, 0, N) {
			num += (int)(L[i] / x);
		}
		return num >= K;
	}

	void solve()
	{
		double lb = 0, ub = INF;
		rep(i, 0, 100) { // 对于浮点数答案，只能尝试一定次数
			double mid = (lb + ub) / 2;
			if (C(mid)) lb = mid;
			else ub = mid;
		}
		printf("%.2f\n", floor(ub * 100) / 100);
	}
}

namespace p142 {
	
	int N, M, x[MAX_N];

	bool C(int d) {
		int last = 0;
		rep(i, 1, M) {
			int crt = last + 1;
			while (crt < N && x[crt] - x[last] < d) crt++;
			if (crt == N) return false;
			last = crt;
		}
		return true;
	}

	void solve()
	{
		sort(x, x + N);
		int lb = 0, ub = INF;
		while (ub - lb > 1) {
			int mid = (ub + lb) / 2;
			if (C(mid)) lb = mid; // 这里实质是最大化答案,这里的答案指的是两牛之间的距离
			else ub = mid;
		}
		cout << lb;
	}
}

namespace p143 {

	int n, k;
	int w[MAX_N], v[MAX_N];
	double y[MAX_N];

	bool C(double x) {
		rep(i, 0, n)
			y[i] = v[i] - x * w[i];
		sort(y, y + n);
		double sum = 0;
		rep(i, 0, k)
			sum += y[n - i - 1];
		return sum >= 0;
	}

	void solve()
	{
		double lb = 0, ub = INF;
		rep(i, 0, 100) {
			double mid = (lb + ub) / 2;
			if (C(mid)) lb = mid;
			else ub = mid;
		}
		printf("%.2f\n", ub);
	}
}

namespace p146 {

	// 给定长度为n的数列a，整数S。找出最短的总和不小于S的连续子序列。
	int n, S;
	int a[MAX_N];

	void solve()
	{
		int res = n + 1;
		int s = 0, t = 0, sum = 0;
		for (;;) {
			while (t < n && sum < S) sum += a[t++];
			if (sum < S) break;
			res = min(res, t - s);
			sum -= a[s++];
		}
		if (res > n) res = 0;
		cout << res;
	}
}

namespace p149 {

	int P;
	int a[MAX_P];

	void solve()
	{
		set<int> all;
		rep(i, 0, P) all.insert(a[i]);
		int n = all.size();

		int s = 0, t = 0, num = 0;
		map<int, int> count;
		int res = P;
		for (;;) {
			while (t < P && num < n)
				if (count[a[t++]]++ == 0)
					num++;
			if (num < n) break;
			res = min(res, t - s);
			if (--count[a[s++]] == 0)
				num--;
		}
		cout << res;
	}
}

namespace p150 {
	
	int N;
	int dir[MAX_N]; // 牛的方向(0代表F,1代表B)

	int f[MAX_N]; //f[i]表示区间[i,i+K-1]是否反转了

	int calc(int K) {
		fill(f, f + N, 0);
		int res = 0;
		int sum = 0;
		rep(i, 0, N - K + 1) {
			if ((dir[i] + sum) % 2 != 0) {
				res++;
				f[i] = 1;
			}
			sum += f[i];
			if (i - K + 1 >= 0)
				sum += f[i - K + 1];
		}
		rep(i, N - K + 1, N) {
			if ((dir[i] + sum) % 2 != 0) {
				return -1;
			}
			if (i - K + 1 >= 0)
				sum -= f[i - K + 1];
		}

		return res;
	}

	void solve()
	{
		int K = 1, M = N;
		rep(k, 1, N + 1) {
			int m = calc(k);
			if (m >= 0 && M > m) {
				M = m;
				K = k;
			}
		}
		cout << K << " " << M << endl;
	}
}

namespace p153 {
	const int dx[5] = { -1,0,0,0,1 };
	const int dy[5] = { 0,-1,0,1,0 };

	int M, N;
	int tile[MAX_M][MAX_N];
	int opt[MAX_M][MAX_N]; // 最优解
	int flip[MAX_M][MAX_N]; // 中间结果

	int get(int x, int y)
	{
		int c = tile[x][y];
		rep(d, 0, 5) {
			int x2 = x + dx[d], y2 = y + dy[d];
			if (x2 >= 0 && x2 < M && y2 >= 0 && y2 < N)
				c += flip[x2][y2];
		}
		return c % 2;
	}

	int calc()
	{
		rep(i, 1, M)
			rep(j, 0, N)
			if (get(i - 1, j) != 0)
				flip[i][j] = 1;

		rep(j, 0, N)
			if (get(M - 1, j) != 0)
				return -1;

		int res = 0;
		rep(i, 0, M)
			rep(j, 0, N)
			res += flip[i][j];

		return res;
	}

	void solve()
	{
		int res = -1;
		rep(i, 0, 1 << N) {
			memset(flip, 0, sizeof(flip));
			rep(j, 0, N)
				flip[0][N - j - 1] = i >> j & 1;
			int num = calc();
			if (num >= 0 && (res < 0 || res > num)) {
				res = num;
				memcpy(opt, flip, sizeof(flip));
			}
		}

		if (res < 0)
			printf("IMPOSSIBLE\n");
		else {
			rep(i, 0, M)
				rep(j, 0, N)
				printf("%d%c", opt[i][j], j + 1 == N ? '\n' : ' ');
		}
	}
}

namespace p158 {
	const double g = 10.0;

	int N, H, R, T;
	double y[MAX_N];

	// 求出T时刻球的位置
	double calc(int T) {
		if (T < 0) return H;
		double t = sqrt(2 * H / g);
		int k = (int)(T / t);
		if (k % 2 == 0) {
			double d = T - k * t;
			return H - g * d * d / 2;
		}
		else {
			double d = (k * t + t - T);
			return H - g * d * d / 2;
		}
	}

	void solve()
	{
		rep(i, 0, N)
			y[i] = calc(T - i);
		sort(y, y + N);
		rep(i, 0, N)
			printf("%.2f%c", y[i] + 2 * R * i / 100.0, i == N - 1 ? '\n' : ' ');
	}
}

namespace p160 {
	int n;
	int A[MAX_N], B[MAX_N], C[MAX_N], D[MAX_N];
	int CD[MAX_N * MAX_N];

	void solve()
	{
		rep(i, 0, n)
			rep(j, 0, n)
			CD[i * n + j] = C[i] + D[j];
		sort(CD, CD + n * n);
		long long res = 0;
		rep(i, 0, n)
			rep(j, 0, n) {
			int cd = -(A[i] + B[j]);
			res += upper_bound(CD, CD + n * n, cd) - lower_bound(CD, CD + n * n, cd);
		}

		cout << res << endl;
	}
}

namespace p162 {

	int n;
	LL w[MAX_N], v[MAX_N];
	LL W;

	pair<LL, LL> ps[1 << (MAX_N / 2)];

	void solve()
	{
		int n2 = n / 2;
		rep(i, 0, (1 << n2)) {
			LL sw = 0, sv = 0;
			rep(j, 0, n2) {
				if (i >> j & 1) {
					sw += w[j];
					sv += v[j];
				}
			}
			ps[i] = make_pair(sw, sv);
		}

		//去除多余的元素
		sort(ps, ps + (1 << n2));
		int m = 1;
		rep(i, 1, 1 << n2) {
			// 保证剩下的元素价值递增
			if (ps[m - 1].second < ps[i].second)
				ps[m++] = ps[i];
		}

		LL res = 0;
		rep(i, 0, 1 << (n - n2)) {
			LL sw = 0, sv = 0;
			rep(j, 0, n - n2) {
				if (i >> j & 1) {
					sw += w[n2 + j];
					sv += v[n2 + j];
				}
			}
			if (sw <= W) {
				LL tv = (lower_bound(ps, ps + m, make_pair(W - sw, (LL)INF)) - 1)->second;
				res = max(res, sv + tv);
			}
		}

		cout << res;
	}
}

namespace p164 {
	const int dx[4] = { 1,0,-1,0 }, dy[4] = { 0,1,0,-1 };

	int W, H, N;
	int X1[MAX_N], X2[MAX_N], Y1[MAX_N], Y2[MAX_N];
	bool fld[MAX_N * 6][MAX_N * 6];

	int compress(int* x1, int* x2, int w) {
		vector<int> xs;
		rep(i, 0, N) {
			rep(d, -1, 2) {
				int tx1 = x1[i] + d, tx2 = x2[i] + d;
				if (tx1 >= 1 && tx1 <= w) xs.push_back(tx1);
				if (tx2 >= 1 && tx2 <= w) xs.push_back(tx2);
			}
		}

		// main模板中的sort_unique就是用来进行离散化的
		sort(xs.begin(), xs.end());
		xs.erase(unique(xs.begin(), xs.end()), xs.end());

		rep(i, 0, N) {
			x1[i] = find(xs.begin(), xs.end(), x1[i]) - xs.begin();
			x2[i] = find(xs.begin(), xs.end(), x2[i]) - xs.begin();
		}

		return xs.size();
	}

	void solve()
	{
		W = compress(X1, X2, W);
		H = compress(Y1, Y2, H);

		memset(fld, 0, sizeof(fld));
		rep(i, 0, N)
			rep(y, Y1[i], Y2[i] + 1)
			rep(x, X1[i], X2[i] + 1)
			fld[y][x] = true;

		int ans = 0;
		rep(y, 0, H)
			rep(x, 0, W) {
			if (fld[y][x]) continue;
			ans++;

			queue<pair<int, int> > que;
			que.push({ x,y });
			while (!que.empty()) {
				int sx = que.front().first, sy = que.front().second;
				que.pop();
				rep(i, 0, 4) {
					int tx = sx + dx[i], ty = sy + dy[i];
					if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
					if (fld[ty][tx]) continue;
					que.push({ tx,ty });
					fld[ty][tx] = true;
				}
			}
		}

		cout << ans << endl;
	}
}

namespace p167 {

	// 这种线段树的写法表现了线段树的实质，即数组中的一个索引实际对应于一个线段，数组中保存的是线段需要维护的信息。
	// 我并不喜欢这种写法，我更喜欢将线段树封装到一个类中。这里为了使原书代码保持一致，照搬了原书的写法。

	int n, dat[2 * MAX_N - 1];

	void init(int n_) {
		n = 1;
		while (n < n_) n *= 2; // 把元素个数扩大到2的幂
		rep(i, 0, 2 * n - 1) dat[i] = INT_MAX;
	}

	void update(int k, int a) {
		k += n - 1;
		dat[k] = a;
		while (k > 0) {
			k = (k - 1) / 2;
			dat[k] = min(dat[k * 2 + 1], dat[k * 2 + 2]);
		}
	}

	int query(int a, int b, int k, int l, int r) {
		if (r <= a || b <= l) return INT_MAX;
		if (a <= l && r <= b) return dat[k];
		else {
			int vl = query(a, b, k * 2 + 1, l, (l + r) / 2);
			int vr = query(a, b, k * 2 + 2, l, (l + r) / 2);
			return min(vl, vr);
		}
	}
}

namespace p170 {
	const double PI = acos(-1.0);
	const int ST_SIZE = (1 << 15) - 1;

	int N, C;
	int L[MAX_N];
	int S[MAX_C], A[MAX_N];

	double vx[ST_SIZE], vy[ST_SIZE];
	double ang[ST_SIZE];

	double prv[MAX_N]; // prv[s]是s相对于s-1的角度

	void init(int k, int l, int r) {
		ang[k] = vx[k] = 0;
		if (l + 1 == r) {
			vy[k] = L[l];
		}
		else {
			int chl = 2 * k + 1, chr = 2 * k + 2;
			init(chl, l, (l + r) / 2);
			init(chr, (l + r) / 2, r);
			vy[k] = vy[chl] + vy[chr];
		}
	}

	void change(int s, double a, int v, int l, int r) {
		if (s <= l) return;
		else if (s < r) {
			int chl = 2 * v + 1, chr = 2 * v + 2;
			int m = (l + r) / 2;
			change(s, a, chl, l, m);
			change(s, a, chr, m, r);
			if (s <= m) ang[v] += a;
			/*
			* 向量(x,y)逆时针绕起点旋转rad度后得到的向量为
			* newx = x*cos(rad)-y*sin(rad)   newy = x*sin(rad)+y*cos(rad)
			*/
			double s = sin(ang[v]), c = cos(ang[v]);
			vx[v] = vx[chl] + (c * vx[chr] - s * vy[chr]);
			vy[v] = vy[chl] + (s * vx[chr] + c * vy[chr]);
		}
	}

	void solve() {
		init(0, 0, N);
		rep(i, 1, N) prv[i] = PI;
		rep(i, 0, C) {
			int s = S[i];
			double a = A[i] / 360.0 * 2 * PI;
			change(s, a - prv[s], 0, 0, N);
			prv[s] = a;
			printf("%.2f %.2f\n", vx[0], vy[0]);
		}
	}
}

namespace p177 {
	LL sum(LL* bit, int i) { LL s = 0; while (i > 0) s += bit[i], i -= (i & -i); return s; }
	void add(LL* bit, int n, int i, int v) { while (i <= n) bit[i] += v, i += (i & -i); }
}

namespace p179 {

	const int DAT_SIZE = (1 << 18) - 1;

	int N, Q;
	int A[MAX_N];
	char T[MAX_Q];
	int L[MAX_Q], R[MAX_Q], X[MAX_Q];
	LL dat_a[DAT_SIZE], dat_b[DAT_SIZE];

	void add(int a, int b, int x, int k, int l, int r) {
		if (b <= l || a >= r) return; // [l,r)不在[a,b)中
		else if (a <= l && r <= b) { // [l,r)被包在[a,b)中
			dat_a[k] += x; // 这个节点整个加上x
			return;
		}
		dat_b[k] += (min(b, r) - max(a, l)) * x; // [l,r)和[a,b)有交集,计算这个节点实际增加的值
		add(a, b, x, 2 * k + 1, l, (l + r) / 2);
		add(a, b, x, 2 * k + 2, (l + r) / 2, r);
	}

	LL sum(int a, int b, int k, int l, int r) {
		if (b <= l || a >= r) return 0; // [l,r)不在[a,b)中
		else if (a <= l && r <= b) // [l,r)被包在[a,b)中
			return dat_a[k] * (r - l) + dat_b[k];

		LL res = (min(b, r) - max(a, l)) * dat_a[k];// [l,r)和[a,b)有交集
		res += sum(a, b, 2 * k + 1, l, (l + r) / 2);
		res += sum(a, b, 2 * k + 2, (l + r) / 2, r);

		return res;
	}

	void solve()
	{
		rep(i, 0, N) add(i, i + 1, A[i], 0, 0, N);
		rep(i, 0, Q) {
			if (T[i] == 'C')
				add(L[i], R[i] + 1, X[i], 0, 0, N);
			else
				cout << sum(L[i], R[i] + 1, 0, 0, N) << endl;
		}
	}
}

namespace p181 {

	int N, Q;
	int A[MAX_N];
	char T[MAX_Q];
	int L[MAX_Q], R[MAX_Q], X[MAX_Q];

	LL bit0[MAX_N + 1], bit1[MAX_N + 1]; // [1,n]
	using namespace p177;

	void solve()
	{
		rep(i, 1, N + 1)
			add(bit0, N, i, A[i]);
		rep(i, 0, Q) {
			if (T[i] == 'C') {
				add(bit0, N, L[i], -X[i] * (L[i] - 1));
				add(bit1, N, L[i], X[i]);
				add(bit0, N, R[i] + 1, X[i] * R[i]);
				add(bit1, N, R[i] + 1, -X[i]);
			}
			else {
				LL res = 0;
				res += sum(bit0, R[i]) + sum(bit1, R[i]) * R[i];
				res -= sum(bit0, L[i] - 1) + sum(bit1, L[i] - 1) * (L[i] - 1);
				cout << res << endl;
			}
		}
	}
}

namespace p185 {

	const int B = 1000;

	int N, M;
	int A[MAX_N];
	int I[MAX_M], J[MAX_M], K[MAX_M];

	int nums[MAX_N];
	vector<int> bucket[MAX_N / B];

	void solve()
	{
		rep(i, 0, N) {
			bucket[i / B].push_back(A[i]);
			nums[i] = A[i];
		}
		sort(nums, nums + N);
		rep(i, 0, N / B) sort(bucket[i].begin(), bucket[i].end());
		rep(i, 0, M) {
			int l = I[i], r = J[i] + 1, k = K[i];

			int lb = -1, ub = N - 1;
			while (ub - lb > 1) {
				int md = (ub + lb) / 2;
				int x = nums[md];

				int tl = l, tr = r, c = 0;
				while (tl < tr && tl % B != 0) if (A[tl++] <= x) c++;
				while (tl < tr && tr % B != 0) if (A[--tr] <= x) c++;

				while (tl < tr) {
					int b = tl / B;
					c += upper_bound(bucket[b].begin(), bucket[b].end(), x)
						- bucket[b].begin();
					tl += B;
				}

				if (c >= k) ub = md;
				else lb = md;
			}

			cout << nums[ub] << endl;
		}
	}
}

namespace p188 {
	const int ST_SIZE = (1 << 18) - 1;

	int N, M;
	int A[MAX_N];
	int I[MAX_M], J[MAX_M], K[MAX_M];

	int nums[MAX_N];
	vector<int> dat[ST_SIZE];

	void init(int k, int l, int r) {
		if (r - 1 == l)
			dat[k].push_back(A[l]);
		else {
			int chl = k * 2 + 1, chr = k * 2 + 2;
			init(chl, l, (l + r) / 2);
			init(chr, (l + r) / 2, r);
			dat[k].resize(r - l);
			merge(all(dat[chl]), all(dat[chr]), dat[k].begin());
		}
	}

	int query(int a, int b, int x, int k, int l, int r) {
		if (r <= a || b <= l) return 0;
		if (a <= l && r <= b) return upper_bound(all(dat[k]), x) - dat[k].begin();
		else {
			int vl = query(a, b, x, k * 2 + 1, l, (l + r) / 2);
			int vr = query(a, b, x, k * 2 + 2, (l + r) / 2, r);
			return vl + vr;
		}
	}


	void solve()
	{
		rep(i, 0, N) nums[i] = A[i];
		sort(nums, nums + N);

		init(0, 0, N);

		rep(i, 0, M) {
			int l = I[i], r = J[i] + 1, k = K[i];

			int lb = -1, ub = N - 1;
			while (ub - lb > 1) {
				int md = (ub + lb) / 2;
				int c = query(l, r, nums[md], 0, 0, N);
				if (c >= k) ub = md;
				else lb = md;
			}

			cout << nums[ub] << endl;
		}
	}
}

namespace p191 {
	int n;
	int d[MAX_N][MAX_N];

	int dp[1 << MAX_N][MAX_N];

	int rec(int S, int v) {
		if (dp[S][v] >= 0)
			return dp[S][v];
		if (S == (1 << n) - 1 && v == 0)
			return dp[S][v] = 0;
		int res = INF;
		rep(u, 0, n) {
			if (!(S >> u & 1))
				res = min(res, rec(S | 1 << u, u) + d[v][u]);
		}
		return dp[S][v] = res;
	}

	void solve()
	{
		memset(dp, -1, sizeof(dp));
		cout << rec(0, 0);
	}
}

namespace p193 {

	int n, m, a, b;
	int t[MAX_N];
	int d[MAX_M][MAX_M];

	double dp[1 << MAX_N][MAX_M];

	void solve()
	{
		rep(i, 0, 1 << n)
			fill(dp[i], dp[i] + m, INF);

		dp[(1 << n) - 1][a] = 0;
		double res = INF;
		for (int S = (1 << n) - 1; S > -1; S--) {
			res = min(res, dp[S][b]);
			rep(v, 0, m)
				rep(i, 0, n)
				if (S >> i & 1)
					rep(u, 0, m)
					if (d[v][u] >= 0)
						dp[S & ~(1 << i)][u] = min(dp[S & ~(1 << i)][u],
							dp[S][v] + (double)d[v][u] / t[i]);
		}
		if (res == INF)
			cout << "Impossible" << endl;
		else
			printf("%.3f\n", res);
	}
}

namespace p196 {

	int n, m, M;
	int color[MAX_N][MAX_M];
	int dp[2][1 << MAX_M];

	void solve()
	{
		int* crt = dp[0], * next = dp[1];
		crt[0] = 1;
		for (int i = n - 1; i > -1; i--)
			for (int j = m - 1; j > -1; j--) {
				for (int used = 0; used < 1 << m; used++) {
					if ((used >> j & 1) || color[i][j])
						next[used] = crt[used & ~(1 << j)];
					else {
						int res = 0;
						if (j + 1 < m && !(used >> (j + 1) & 1) && !color[i][j + 1])
							res += crt[used | 1 << (j + 1)];
						if (i + 1 < n && !color[i + 1][j])
							res += crt[used | 1 << j];
						next[used] = res % M;
					}
				}
				swap(crt, next);
			}

		cout << crt[0] << endl;
	}
}

namespace p199_part1 {

	mat operator*(const mat& a, const mat& b)
	{
		const int M = 10000;
		mat c(a.size(), vec(b[0].size()));
		for (int i = 0; i < a.size(); i++)
			for (int k = 0; k < b.size(); k++)
				for (int j = 0; j < b[0].size(); j++)
					c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % M;
		return c;
	}

	mat pow(mat a, LL n)
	{
		mat b(a.size(), vec(a.size()));
		for (int i = 0; i < a.size(); i++)
			b[i][i] = 1;
		while (n)
		{
			if (n & 1) b = b * a;
			a = a * a;
			n >>= 1;
		}
		return b;
	}
}

namespace p199_part2 {

	using namespace p199_part1;

	LL n;

	void solve()
	{
		mat A(2, vec(2));
		A = { {1,1},
			 {1,0} };
		A = pow(A, n);
		cout << A[1][0];
	}
}

namespace p202 {

	using namespace p199_part1;

	int n;

	void solve()
	{
		mat A(3, vec(3));
		A = { {2,1,0},
			 {2,2,2},
			 {0,1,2} };
		A = pow(A, n);
		cout << A[0][0] << endl;
	}
}

namespace p203 {

	using namespace p199_part1;

	int n, k;
	int g[MAX_N][MAX_N];

	void solve()
	{
		mat A(n, vec(n));
		rep(i, 0, n)
			rep(j, 0, n)
			if (g[i][j] == 1) A[i][j] = 1;
		A = pow(A, k);
		int res = 0;
		rep(i, 0, n)
			rep(j, 0, n)
			if (A[i][j] != 0)
				res++;
		cout << res;
	}
}

namespace p204 {

	int n, k, M;
	mat A;

	void solve()
	{
		mat B(n * 2, vec(n * 2));
		rep(i, 0, n) {
			rep(j, 0, n)
				B[i][j] = A[i][j];
			B[n + i][i] = B[n + i][n + i] = 1;
		}
		B = pow(B, k + 1);
		rep(i, 0, n)
			rep(j, 0, n) {
			int a = B[n + i][j] % M;
			if (i == j) a = (a + M - 1) % M;
			printf("%d%c", a, j + 1 == n ? '\n' : ' ');
		}
	}
}

namespace p209 {
	vector<edge> G[MAX_V];
	bool used[MAX_V];

	void add_edge(int from, int to, int cap) {
		G[from].push_back({ to, cap, (int)G[to].size() });
		G[to].push_back({ from, 0, (int)G[from].size() - 1 });
	}

	int dfs(int v, int t, int f) {
		if (v == t) return f;
		used[v] = true;
		rep(i, 0, G[v].size()) {
			edge& e = G[v][i];
			if (!used[e.to] && e.cap > 0) {
				int d = dfs(e.to, t, min(f, e.cap));
				if (d > 0) {
					e.cap -= d;
					G[e.to][e.rev].cap += d;
					return d;
				}
			}
		}
		return 0;
	}

	int max_flow(int s, int t) {
		int flow = 0;
		for (;;) {
			memset(used, 0, sizeof(used));
			int f = dfs(s, t, INF);
			if (f == 0) return flow;
			flow += f;
		}
	}
}

namespace p216 {

	vector<edge> G[MAX_V];
	int level[MAX_V];
	int iter[MAX_V];

	void add_edge(int from, int to, int cap) {
		G[from].push_back({ to, cap, (int)G[to].size() });
		G[to].push_back({ from, 0, (int)G[from].size() - 1 });
	}

	int bfs(int s) {
		memset(level, -1, sizeof(level));
		queue<int> que;
		level[s] = 0;
		que.push(s);
		while (!que.empty()) {
			int v = que.front(); que.pop();
			rep(i, 0, G[v].size()) {
				edge& e = G[v][i];
				if (e.cap > 0 && level[e.to] < 0) {
					level[e.to] = level[v] + 1;
					que.push(e.to);
				}
			}
		}
		return 0;
	}

	int dfs(int v, int t, int f) {
		if (v == t) return f;
		for (int& i = iter[v]; i < G[v].size(); i++) {
			edge& e = G[v][i];
			if (e.cap > 0 && level[v] < level[e.to]) {
				int d = dfs(e.to, t, min(f, e.cap));
				if (d > 0) {
					e.cap -= d;
					G[e.to][e.rev].cap += d;
					return d;
				}
			}
		}
		return 0;
	}

	int max_flow(int s, int t) {
		int flow = 0;
		for (;;) {
			bfs(s);
			if (level[t] < 0) return flow;
			memset(iter, 0, sizeof(iter));
			int f;
			while ((f = dfs(s, t, INF)) > 0)
				flow += f;
		}
	}
}

namespace p219 {

	int V;
	vector<int> G[MAX_V];
	int match[MAX_V];
	bool used[MAX_V];

	bool dfs(int v) {
		used[v] = true;
		rep(i, 0, G[v].size()) {
			int u = G[v][i], w = match[u];
			if (w < 0 || !used[w] && dfs(w)) {
				match[v] = u;
				match[u] = v;
				return true;
			}
		}
		return false;
	}

	void add_edge(int u, int v) {
		G[u].push_back(v);
		G[v].push_back(u);
	}

	int biparite_matching() {
		int res = 0;
		memset(match, -1, sizeof(match));
		rep(v, 0, V) {
			if (match[v] < 0) {
				memset(used, 0, sizeof(used));
				if (dfs(v))
					res++;
			}
		}
		return res;
	}

}

namespace p222 {

	int V;
	vector<edge> G[MAX_V];
	int dist[MAX_V];
	int prevv[MAX_V], preve[MAX_V];

	void add_edge(int from, int to, int cap, int cost) {
		G[from].push_back({ to, cap, cost, (int)G[to].size() });
		G[to].push_back({ from, 0, -cost, (int)G[from].size() - 1 });
	}

	int min_cost_flow(int s, int t, int f) {
		int res = 0;
		while (f > 0) {
			fill(dist, dist + V, INF);
			dist[s] = 0;
			bool update = true;
			while (update) {
				update = false;
				rep(v, 0, V) {
					if (dist[v] == INF) continue;
					rep(i, 0, G[v].size()) {
						edge& e = G[v][i];
						if (e.cap > 0 && dist[e.to] > dist[v] + e.cost) {
							dist[e.to] = dist[v] + e.cost;
							prevv[e.to] = v;
							preve[e.to] = i;
							update = true;
						}
					}
				}
			}

			if (dist[t] == INF)
				return -1;

			int d = f;
			for (int v = t; v != s; v = prevv[v]) {
				d = min(d, G[prevv[v]][preve[v]].cap);
			}
			f -= d;
			res += d * dist[t];
			for (int v = t; v != s; v = prevv[v]) {
				edge& e = G[prevv[v]][preve[v]];
				e.cap -= d;
				G[v][e.rev].cap += d;
			}
		}

		return res;
	}
}

namespace p228 {
	int N, K;
	int R[MAX_K], C[MAX_K];

	using namespace p219;

	void solve()
	{
		V = N * 2;
		rep(i, 0, K)
			add_edge(R[i] - 1, N + C[i] - 1);
		cout << biparite_matching() << endl;
	}
}

namespace p230 {
	const int dx[4] = { -1,0,0,1 };
	const int dy[4] = { 0,-1,1,0 };

	int X, Y;
	char field[MAX_X][MAX_Y + 1];
	vector<int> dX, dY;
	vector<int> pX, pY;
	int dist[MAX_X][MAX_Y][MAX_X][MAX_Y];

	using namespace p219;

	void bfs(int x, int y, int d[MAX_X][MAX_Y])
	{
		queue<int> qx, qy;
		d[x][y] = 0;
		qx.push(x);
		qy.push(y);
		while (!qx.empty()) {
			x = qx.front(); qx.pop();
			y = qy.front(); qy.pop();
			rep(k, 0, 4) {
				int x2 = x + dx[k], y2 = y + dy[k];
				if (x2 >= 0 && x2 < X && y2 >= 0 && y2 < Y && field[x2][y2] == '.' && d[x2][y2] < 0) {
					d[x2][y2] = d[x][y] + 1;
					qx.push(x2);
					qy.push(y2);
				}
			}
		}
	}

	void solve()
	{
		int n = X * Y;
		dX.clear(); dY.clear();
		pX.clear(); pY.clear();
		memset(dist, -1, sizeof(dist));

		rep(x, 0, X) {
			rep(y, 0, Y) {
				if (field[x][y] == 'D') {
					dX.push_back(x);
					dY.push_back(y);
					bfs(x, y, dist[x][y]);
				}
				else if (field[x][y] == '.') {
					pX.push_back(x);
					pY.push_back(y);
				}
			}
		}

		memset(G, 0, sizeof(G));
		int d = dX.size(), p = pX.size();
		rep(i, 0, d)
			rep(j, 0, p) {
			if (dist[dX[i]][dY[i]][pX[j]][pY[j]] >= 0) {
				rep(k, dist[dX[i]][dY[i]][pX[j]][pY[j]], n + 1)
					add_edge((k - 1) * d + i, n * d + j); // 门 <-> 人
			}
		}

		if (p == 0) {
			cout << 0 << endl;
			return;
		}

		int num = 0;
		memset(match, -1, sizeof(match));
		rep(v, 0, n * d) {
			memset(used, 0, sizeof(used));
			if (dfs(v)) {
				if (++num == p) {
					cout << v / d + 1 << endl;
					return;
				}
			}
		}

		cout << "impossible" << endl;
	}
}

namespace p234 {
	int N, F, D;
	bool likeF[MAX_N][MAX_F];
	bool likeD[MAX_N][MAX_D];

	using namespace p209;

	void solve()
	{
		/*
		[0,N) 食物一侧的牛
		[N,2N) 饮料一侧的牛
		[2N,2N+F) 食物
		[2N+F,2N+F+D) 饮料
		*/
		int s = N * 2 + F + D, t = s + 1;

		// 在s与食物之间连边
		rep(i, 0, F)
			add_edge(s, N * 2 + i, 1);

		// 在饮料与t之间连边
		rep(i, 0, D)
			add_edge(N * 2 + F + i, t, 1);

		rep(i, 0, N) {
			// 在食物一侧的牛和饮料一侧的牛之间连边
			add_edge(i, N + i, 1);

			// 在牛和所喜欢的食物或饮料之间连边
			rep(j, 0, F)
				if (likeF[i][j]) add_edge(N * 2 + j, i, 1);

			rep(j, 0, D)
				if (likeD[i][j]) add_edge(N + i, N * 2 + F + j, 1);
		}

		cout << max_flow(s, t) << endl;
	}
}

namespace p236 {

	int N, M;
	int A[MAX_N], B[MAX_N];
	int a[MAX_M], b[MAX_M], w[MAX_M];

	using namespace p209;

	void solve()
	{
		int s = N, t = s + 1;
		rep(i, 0, N) {
			add_edge(i, t, A[i]);
			add_edge(s, i, B[i]);
		}
		rep(i, 0, M) {
			add_edge(a[i] - 1, b[i] - 1, w[i]);
			add_edge(b[i] - 1, a[i] - 1, w[i]);
		}

		printf("%d\n", max_flow(s, t));
	}
}

namespace p238 {
	int N, M;
	int a[MAX_M], b[MAX_M], c[MAX_M];

	using namespace p222;

	void solve()
	{
		int s = 0, t = N - 1;
		V = N;
		rep(i, 0, M) {
			add_edge(a[i] - 1, b[i] - 1, 1, c[i]);
			add_edge(b[i] - 1, a[i] - 1, 1, c[i]);
		}

		printf("%d\n", min_cost_flow(s, t, 2));
	}
}

namespace p240 {
	int N, M;
	int X[MAX_N], Y[MAX_N], B[MAX_N];
	int P[MAX_M], Q[MAX_M], C[MAX_M];
	int E[MAX_N][MAX_N];

	using namespace p222;

	void solve()
	{
		// [0,N) 大楼
		// [N,N+M) 防空洞

		int s = N + M, t = s + 1;
		V = t + 1;
		int cost = 0; // 计算避难计划的总花费
		int F = 0; // 总人数
		rep(i, 0, N)
			rep(j, 0, M) {
			int c = abs(X[i] - P[j]) + abs(Y[i] - Q[j]) + 1;
			add_edge(i, N + j, INF, c);
			cost += E[i][j] * c;
		}

		rep(i, 0, N) {
			add_edge(s, i, B[i], 0);
			F += B[i];
		}

		rep(i, 0, M)
			add_edge(N + i, t, C[i], 0);

		if (min_cost_flow(s, t, F) < cost) {
			printf("SUBOPTIMAL\n");
			rep(i, 0, N)
				rep(j, 0, M)
				printf("%d%c", G[N + j][i].cap, j + 1 == M ? '\n' : ' ');
		}
		else {
			printf("OPTIMAL\n");
		}
	}
}

namespace p242 {

	int N, M;
	int X[MAX_N], Y[MAX_N], B[MAX_N];
	int P[MAX_M], Q[MAX_M], C[MAX_M];
	int E[MAX_N][MAX_N];

	void solve() {
		// TODO
	}
}

namespace p243 {
	int N, M;
	int Z[MAX_N][MAX_M];

	using namespace p222;

	void solve()
	{
		/*
		[0,N) 玩具
		[N,2N) 0号工厂
		...
		[MN,(M+1)N) M-1号工厂
		*/
		int s = N + N * M, t = s + 1;
		V = t + 1;
		rep(i, 0, N)
			add_edge(s, i, 1, 0);
		rep(j, 0, M)
			rep(k, 0, N) {
			add_edge(N + j * N + k, t, 1, 0);
			rep(i, 0, N)
				add_edge(i, N + j * N + k, 1, (k + 1) * Z[i][j]);
		}

		printf("%.6lf\n", (double)min_cost_flow(s, t, N) / N);
	}
}

namespace p246 {

	int N, K;
	int a[MAX_N], b[MAX_N], w[MAX_N];

	void solve() {
		// TODO
	}
}

namespace p250_part1 {

	double EPS = 1e-10;

	double add(double a, double b) {
		if (abs(a + b) < EPS * (abs(a) + abs(b))) return 0;
		return a + b;
	}

	P::P() {}
	P::P(double x, double y) : x(x), y(y) {}
	P P::operator+(P p) { return P(add(x, p.x), add(y, p.y)); }
	P P::operator-(P p) { return P(add(x, -p.x), add(y, -p.y)); }
	P P::operator*(double d) { return P(x * d, y * d); }
	double P::dot(P p) { return add(x * p.x, y * p.y); } // 点积
	double P::det(P p) { return add(x * p.y, -y * p.x); } // 叉积

	// 判断点q是否在直线上
	bool on_seg(P p1, P p2, P q) {
		return (p1 - q).det(p2 - q) == 0 && (p1 - q).dot(p2 - q) <= 0;
	}

	// 计算两直线的交点
	P intersection(P p1, P p2, P q1, P q2) {
		return p1 + (p2 - p1) * ((q2 - q1).det(q1 - p1) / (q2 - q1).det(p2 - p1));
	}
}


namespace p250_part2 {

	using namespace p250_part1;

	int n;
	P p[MAX_N], q[MAX_N];
	int m;
	int a[MAX_M], b[MAX_M];

	bool g[MAX_N][MAX_N];

	void solve()
	{
		rep(i, 0, n) {
			g[i][i] = true;
			rep(j, 0, i) {
				if ((p[i] - q[i]).det(p[j] - q[j]) == 0) {
					g[i][j] = g[j][i] = on_seg(p[i], q[i], p[j])
						|| on_seg(p[i], q[i], q[j])
						|| on_seg(p[j], q[j], p[i])
						|| on_seg(p[j], q[j], q[i]);
				}
				else {
					P r = intersection(p[i], q[i], p[j], q[j]);
					g[i][j] = g[j][i] = on_seg(p[i], q[i], r) && on_seg(p[j], q[j], r);
				}
			}
		}

		rep(k, 0, n)
			rep(i, 0, n)
			rep(j, 0, n)
			g[i][j] |= g[i][k] && g[k][j];

		rep(i, 0, m)
			if (g[a[i] - 1][b[i] - 1])
				cout << "CONNECTED" << endl;
			else
				cout << "NOT CONNECTED" << endl;
	}
}

namespace p255 {

	int N, V, X, Y;
	int L[MAX_N], B[MAX_N], R[MAX_N], T[MAX_N];

	void solve() {
		// TODO
	}
}

namespace p258 {
	int N;
	double x[MAX_N], y[MAX_N], r[MAX_N];

	// 判断圆i是否在圆j内部
	bool inside(int i, int j) {
		double dx = x[i] - x[j], dy = y[i] - y[j];
		return dx * dx + dy * dy <= r[j] * r[j];
	}

	void solve()
	{
		vector<pair<double, int> > events;
		rep(i, 0, N) {
			events.push_back(make_pair(x[i] - r[i], i));
			events.push_back(make_pair(x[i] + r[i], i + N));
		}
		sort(events.begin(), events.end());

		set<pair<double, int> > outers; // 与扫描线相交的最外层圆的集合
		vector<int> res;

		rep(i, 0, events.size()) {
			int id = events[i].second % N;
			if (events[i].second < N) {
				set<pair<double, int> >::iterator it = outers.lower_bound(make_pair(y[id], id));
				if (it != outers.end() && inside(id, it->second)) continue;
				if (it != outers.begin() && inside(id, (--it)->second)) continue;
				res.push_back(id);
				outers.insert(make_pair(y[id], id));
			}
			else {
				outers.erase(make_pair(y[id], id));
			}
		}

		sort(res.begin(), res.end());
		printf("%d\n", res.size());
		rep(i, 0, res.size()) {
			printf("%d ", res[i] + 1);
		}
	}
}

namespace p260 {

	bool cmp_x(const P& p, const P& q) {
		if (p.x != q.x) return p.x < q.x;
		return p.y < q.y;
	}

	vector<P> convex_hull(P* ps, int n) {
		sort(ps, ps + n, cmp_x);
		int k = 0;
		vector<P> qs(n * 2);
		rep(i, 0, n) {
			while (k > 1 && (qs[k - 1] - qs[k - 2]).det(ps[i] - qs[k - 1]) <= 0)
				k--;
			qs[k++] = ps[i];
		}
		for (int i = n - 2, t = k; i > -1; i--) {
			while (k > t && (qs[k - 1] - qs[k - 2]).det(ps[i] - qs[k - 1]) <= 0)
				k--;
			qs[k++] = ps[i];
		}
		qs.resize(k - 1);
		return qs;
	}


	double dist(P p, P q) {
		return (p - q).dot(p - q);
	}

	int N;
	P ps[MAX_N];

	void solve()
	{
		vector<P> qs = convex_hull(ps, N);
		int n = qs.size();
		if (n == 2) {
			printf("%.0f\n", dist(qs[0], qs[1]));
			return;
		}

		int i = 0, j = 0;
		rep(k, 0, n) {
			if (!cmp_x(qs[i], qs[k]))
				i = k;
			if (cmp_x(qs[j], qs[k]))
				j = k;
		}

		double res = 0;
		int si = i, sj = j;
		while (i != sj || j != si) {
			res = max(res, dist(qs[i], qs[j]));
			if ((qs[(i + 1) % n] - qs[i]).det(qs[(j + 1) % n] - qs[j]) < 0)
				i = (i + 1) % n;
			else
				j = (j + 1) % n;
		}

		printf("%.0f\n", res);
	}
}

namespace p263 {
	int M, N;
	int X1[MAX_M], Y1[MAX_M];
	int X2[MAX_N], Z2[MAX_N];

	// 计算按x值对多边形切片截得的线段长度
	double clip(int* X, int* Y, int n, double x)
	{
		double lb = INF, ub = -INF;
		rep(i, 0, n) { // 凸多边形的顶点是逆时针给出的
			double x1 = X[i], y1 = Y[i];
			double x2 = X[(i + 1) % n], y2 = Y[(i + 1) % n];
			if ((x1 - x) * (x2 - x) <= 0 && x1 != x2) {
				double y = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
				lb = min(lb, y);
				ub = max(ub, y);
			}
		}
		return max(0.0, ub - lb);
	}

	void solve()
	{
		int min1 = *min_element(X1, X1 + M), max1 = *max_element(X1, X1 + M);
		int min2 = *min_element(X2, X2 + N), max2 = *max_element(X2, X2 + N);
		vector<int> xs;
		rep(i, 0, M) xs.push_back(X1[i]);
		rep(i, 0, N) xs.push_back(X2[i]);
		sort(xs.begin(), xs.end());

		double res = 0;
		rep(i, 0, xs.size() - 1) {
			double a = xs[i], b = xs[i + 1], c = (a + b) / 2;
			double fa = clip(X1, Y1, M, a) * clip(X2, Z2, N, a);
			double fb = clip(X1, Y1, M, b) * clip(X2, Z2, N, b);
			double fc = clip(X1, Y1, M, c) * clip(X2, Z2, N, c);
			res += (b - a) / 6 * (fa + 4 * fc + fb);
		}

		printf("%.10f\n", res);
	}
}

namespace p286 {
	const double EPS = 1e-18;

	vec gauss_jordan(const mat& A, const vec& b) {
		int n = A.size();
		mat B(n, vec(n + 1));

		rep(i, 0, n)
			rep(j, 0, n)
			B[i][j] = A[i][j];
		rep(i, 0, n) B[i][n] = b[i];

		rep(i, 0, n) {
			int pivot = i;
			rep(j, i, n)
				if (abs(B[j][i]) > abs(B[pivot][i])) pivot = j;
			swap(B[i], B[pivot]);

			if (abs(B[i][i]) < EPS) return vec();

			rep(j, i + 1, n + 1) B[i][j] /= B[i][i];
			rep(j, 0, n)
				if (i != j)
					rep(k, i + 1, n + 1)
					B[j][k] -= B[j][i] * B[i][k];
		}
		vec x(n);
		rep(i, 0, n) x[i] = B[i][n];
		return x;
	}
}

namespace p291 {

	using namespace p115;

	int mod_inverse(int a, int m) {
		int x, y;
		extgcd(a, m, x, y);
		return (m + x % m) % m;
	}
}

/*
欧拉函数
*/
namespace p292 {
	int euler_phi(int n) {
		int res = n;
		for (int i = 2; i * i <= n; i++) {
			if (n % i == 0) {
				res = res / i * (i - 1);
				for (; n % i == 0; n /= i);
			}
		}
		if (n != 1)res = res / n * (n - 1);
		return res;
	}

	int euler[MAX_N];

	void euler_phi2() {
		rep(i, 0, MAX_N) euler[i] = i;
		rep(i, 2, MAX_N) {
			if (euler[i] == i)
				for (int j = i; j < MAX_N; j += i)
					euler[j] = euler[j] / i * (i - 1);
		}
	}
}

namespace p293 {
	using namespace p114;
	using namespace p291;

	PII linear_congruence(VI& A, VI& B, VI& M) {
		int x = 0, m = 1;
		rep(i, 0, A.size()) {
			int a = A[i] * m, b = B[i] - A[i] * x, d = gcd(M[i], a);
			if (b % d != 0) return make_pair(0, -1);
			int t = b / d * mod_inverse(a / d, M[i] / d) % (M[i] / d);
			x = x + m * t;
			m *= M[i] / d;
		}
		return make_pair(x % m, m);
	}
}

}
