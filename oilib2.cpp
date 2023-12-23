#include "types.h"
#include "oilib2.h"

namespace lib2 {

namespace p122 {

	const int MAX_N = 200001;
	const int MAX_M = 200001;

	int n, m; // 矿石总数、区间总数和标准值
	LL s;
	int w[MAX_N], v[MAX_N]; // 矿石重量和价值
	int l[MAX_M], r[MAX_M];

	void read_case() {
		read(n, m, s);
		rep(i, 1, n + 1)
			read(w[i], v[i]);
		rep(i, 1, m + 1)
			read(l[i], r[i]);
	}

	LL check(int W) {
		LL y = 0;

		VI cnt(n + 1); // vcnt[i]表示前i个重量超过W的矿石总数
		VLL vsum(n + 1); // vsum[i]表示前i个重量超过W的矿石的价值之和
		rep(i, 1, n + 1) {
			if (w[i] >= W) { cnt[i] = 1; vsum[i] = v[i]; }
			cnt[i] += cnt[i - 1];
			vsum[i] += vsum[i - 1];
		}
		rep(i, 1, m + 1) {
			y += 1LL * (cnt[r[i]] - cnt[l[i] - 1]) * (vsum[r[i]] - vsum[l[i] - 1]);
		}
		return y;
	}

	void solve() {
        int lb = -1, ub = 1000000;
        LL ans = -1;
        while (ub - lb > 1) {
            int W = (lb + ub) / 2;
            LL y = check(W);
            if (y > s) lb = W;
            else ub = W;
            if (ans == -1)ans = abs(s - y);
            else ans = min(ans, abs(s - y));
        }
		print(ans);
	}
}

namespace p123 {

	const int INF = 1000000000;
	const int MAX_N = 50002;

	int L, n, m;
	int d[MAX_N];

	void read_case() {
		read(L, n, m);
		rep(i, 1, n + 1)
			read(d[i]);
		d[n + 1] = L;
	}

	// 能否移动至多M个岩石使得最短跳跃距离大于等于k
	int check(int k) {
		int i = 0, c = 0;
		while (i <= n) {
			int j = i + 1;
			while (j <= n + 1 && d[j] - d[i] < k) {
				j++;
				c++;
			}
			if (d[j] - d[i] < k) {
				return false;
			}
			i = j;
		}
		return c <= m;
	}

	void solve() {
		// 目标是移走至多M个岩石使得最短跳跃距离尽可能地长
		int lb = 0, ub = INF;
		while (ub - lb > 1) {
			int mid = (lb + ub) / 2;
			if (check(mid)) lb = mid;
			else ub = mid;
		}
		cout << lb << endl;
	}
}

namespace p124 {

	const int MAXN = 1000009;
	const int MAXM = 1000009;

	int n, m, r[MAXN];
	int d[MAXM], s[MAXM], t[MAXM];

	void read_case() {
		read(n, m);
		rep(i, 0, n)
			read(r[i]);
		rep(i, 0, m)
			read(d[i], s[i], t[i]);
	}

	bool check(int mm)
	{
		VI a(n);
		rep(i, 0, mm + 1) {
			a[s[i]] += d[i];
			if (t[i] + 1 < n)
				a[t[i] + 1] -= d[i];
		}
		rep(i, 1, n)
			a[i] += a[i - 1];
		rep(i, 0, n)
			if (a[i] > r[i])
				return false;
		return true;
	}

	void solve() {
		rep(i, 0, m) {
			s[i] -= 1;
			t[i] -= 1;
		}
			
		int lo = 0;
		int hi = m - 1;
		int ans = -1;
		while (lo <= hi) {
			int mm = (lo + hi) / 2;
			if (check(mm))
				lo = mm + 1;
			else {
				ans = mm;
				hi = mm - 1;
			}
		}
		if (ans == -1)
			print(0);
		else {
			print(-1);
			print(ans + 1);
		}
	}
}

namespace p125 {

	const int MAX_N = 100000, MAX_M = 100000;

	int n;
	int h[MAX_N];
	int x0;
	int m;
	int s[MAX_M], x[MAX_M];

	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(h[i]);
		read(x0);
		read(m);
		rep(i, 1, m + 1)
			read(s[i], x[i]);
	}

	// 求出从j出发总里程为x的情况下小a和小b的里程总数
	void query(vector<VLL>& jmp, vector<VLL>& f, vector<VLL>& g, int j, int x, LL& da, LL& db) {
		da = db = 0;
		repd(i, 21, -1) {
			if (f[i][j] + g[i][j] <= x) {
				x -= f[i][j] + g[i][j];
				da += f[i][j];
				db += g[i][j];
				j = jmp[i][j];
			}
		}
		if (f[0][j] <= x) {
			x -= f[0][j];
			da += f[0][j];
		}
	}

	void solve() {
		// 以下开一轮指a,b各开一次

		set<PII> S;
		VI a(n + 1), b(n + 1);

		repd(i, n, 0) { // 找到小a和小b的从i出发的目的地
			S.insert({ h[i],i });

			vector<pair<int, PII>> v; //{距离,{海拔高度,城市id}}
			set<PII>::iterator pre, succ;
			pre = succ = S.lower_bound({ h[i],i });
			if (pre != S.begin()) {
				v.push_back({ 0,*(--pre) }); // 不把h[i]本身算进去
				if (pre != S.begin())
					v.push_back({ 0,*(--pre) });
			}
			if (++succ != S.end()) { // 不把h[i]本身算进去
				v.push_back({ 0,*succ });
				if (++succ != S.end()) {
					v.push_back({ 0,*succ });
				}
			}
			rep(j, 0, v.size())
				v[j].first = abs(v[j].second.first - h[i]);
			sort(all(v));
			if (v.size() > 0)b[i] = v[0].second.second;
			if (v.size() > 1)a[i] = v[1].second.second;


		}

		//	W(range(0,n+1));
		//    W(a);
		//    W(b);

		vector<VLL> jmp, f, g; // 计算第一轮的状态
		jmp = f = g = vector<VLL>(22, VLL(n + 1));
		rep(i, 1, n + 1) {
			jmp[0][i] = b[a[i]]; // 计算从i开始开一轮到达的城市
			if (a[i]) f[0][i] = abs(h[a[i]] - h[i]);
			if (b[a[i]] && a[i]) g[0][i] = abs(h[b[a[i]]] - h[a[i]]);
		}

		//    W("round",0);
		//    W(range(0,n+1));
		//    W(f[0]);
		//    W(g[0]);


			// 计算经过2**i轮的状态
		rep(i, 1, 22) {
			rep(j, 1, n + 1) {
				jmp[i][j] = jmp[i - 1][jmp[i - 1][j]];
				f[i][j] = f[i - 1][j] + f[i - 1][jmp[i - 1][j]];
				g[i][j] = g[i - 1][j] + g[i - 1][jmp[i - 1][j]];
			}
			//    	W("round",i);
			//    	W(range(0,n+1));
			//    	W(f[i]);
			//    	W(g[i]);
		}


		LL da_ = 1, db_ = 0, i_, h_, da, db;
		rep(i, 1, n + 1) {
			query(jmp, f, g, i, x0, da, db);
			if (da * db_ < db * da_) {
				i_ = i;
				h_ = h[i];
				da_ = da;
				db_ = db;
			}
		}
		cout << i_ << endl;

		rep(i, 1, m + 1) {
			query(jmp, f, g, s[i], x[i], da, db);
			cout << da << " " << db << endl;
		}
	}
}

namespace p126 {

	const int MAXN = 100000;
	const int MAXM = 100000;
	const int MAXK = 100000;

	typedef struct EDGE
	{
		int from, to, next;
	}EDGE, * EDGE_PTR;

	int n, m, k; // n:城市数 m:边数 k:敌城数
	int a[MAXK]; // 无效顶点
	int U[MAXM], V[MAXM];

	void read_case() {
		read(n, m, k);
		rep(i, 0, k)
			read(a[i]);
		rep(i, 0, m)
			read(U[i], V[i]);
	}

	VI valid;
	vector<VI> adj, adjv;
	vector<VI> cnt;

	VI S, S1;
	vector<VI> cnt_;

	void del(int i, double c) {
		queue<int> Q;
		Q.push(i);
		while (Q.size() > 0) {
			int u = Q.front(); Q.pop();
			S[u] = 0;
			for (int v : adjv[u]) {
				cnt_[v][0] -= 1;
				double t = (double)cnt_[v][0] / (double)cnt_[v][1];
				if (t < c && S[v] == 1)
					Q.push(v);
			}
		}
	}

	bool check(double c) {
		cnt_ = cnt;
		S = VI(n, 1);
		S1.clear();
		rep(i, 0, n) {
			if (valid[i] == 1)
				if (S[i] == 1) {
					double t = (double)cnt_[i][0] / (double)cnt_[i][1];
					if (t < c)
						del(i, c);
				}
			else
				S[i] = 0;
		}
		rep(i, 0, n)
			if (S[i] == 1)
				S1.push_back(i + 1);
		if (S1.size() > 0)
			return true;
		else
			return false;
	}

	void solve() {
		valid = VI(n, 1);
		rep(i, 0, k)
			valid[a[i] - 1] = 0;
		adj = vector<VI>(n);
		adjv = vector<VI>(n);
		rep(i, 0, m) {
			int u = U[i]-1;
			int v = V[i]-1;
			adj[u].push_back(v);
			adj[v].push_back(u);
			if (valid[v] == 1)
				adjv[u].push_back(v);
			if (valid[u] == 1)
				adjv[v].push_back(u);
		}
		cnt = vector<VI>(n, VI(2));
		rep(u, 0, n) {
			cnt[u][0] = adjv[u].size(); // 友方邻居数
			cnt[u][1] = adj[u].size(); // 总邻居数
		}
		
		double lo = 0, hi = 1;
		VI ans;
		rep(i, 0, 100) {
			double c = (lo + hi) / 2;
			if (check(c)) {
				ans = S1;
				lo = c;
			}
			else
				hi = c;
		}
		print(ans.size());
		print(ans);
	}
}

namespace p127 {
	const int MAXN = 300000;
	int n, a[MAXN];
	
	void read_case() {
		read(n);
		rep(i, 1, n+1)
			read(a[i]);
	}

	int gcd(int a, int b) {
		if (b == 0) return a;
		else return gcd(b, a % b);
	}

	// 检查是否存在长为k的符合要求的区间
	int check(int n, vector<VI>& f, vector<VI>& g, int mid) {
		rep(i, 1, n + 1) {
			int j = i + mid;
			if (j <= n + 1) {
				int k = log2(j - i);
				if (min(f[k][i], f[k][j - (1 << k)]) == gcd(g[k][i], g[k][j - (1 << k)]))
					return true;
			}
			else
				return false;

		}
		return false;
	}

	void solve() {
		vector<VI> f, g; // f保存最小值,g保存gcd
		f = g = vector<VI>(20, VI(n + 1));
		rep(i, 1, n + 1) {
			f[0][i] = a[i];
			g[0][i] = a[i];
		}
		//    W(0,':');
		//    W(f[0]);
		//    W(g[0]);

		rep(i, 1, 20) {
			int k = 1 << (i - 1);
			rep(j, 1, n + 1 - k) {
				f[i][j] = min(f[i - 1][j], f[i - 1][j + k]);
				g[i][j] = gcd(g[i - 1][j], g[i - 1][j + k]);
			}
			//    	W(i,':');
			//    	W(f[i]);
			//    	W(g[i]);
		}

		int lb = 0, ub = n + 1;
		while (ub - lb > 1) {
			int mid = (ub + lb) / 2;
			if (check(n, f, g, mid)) lb = mid;
			else ub = mid;
		}

		//W(lb);

		VI ans;
		rep(i, 1, n + 1) {
			int j = i + lb;
			if (j <= n + 1) {
				int k = log2(j - i);
				if (min(f[k][i], f[k][j - (1 << k)]) == gcd(g[k][i], g[k][j - (1 << k)]))
					ans.push_back(i);
			}
			else
				break;
		}
		print(ans.size(), lb - 1);
		print(ans);
	}
}

namespace p128 {

	const int MAXN = 100009;

	int n, a[MAXN];
	
	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(a[i]);
	}
	
	LL sum[MAXN];
	struct POINT { LL x, y; } point[MAXN];
	int ptot;

	void add_point(LL x, LL y)
	{
		POINT pt;
		pt.x = x;
		pt.y = y;
		point[++ptot] = pt;
	}

	// 定义一些工具函数
	bool cmpy(const int& a, const int& b) { return point[a].y < point[b].y; }
	LL dist(POINT* p1, POINT* p2) { return (p2->x - p1->x) * (p2->x - p1->x) + (p2->y - p1->y) * (p2->y - p1->y); }

	int tmp[MAXN];
	LL work(int ql, int qr)
	{
		LL d, d1, d2;
		int k;
		d = 1e18;
		if (ql == qr)
			return d;
		if (ql + 1 == qr)
			return dist(&point[ql], &point[qr]);
		int mid = (ql + qr) / 2;
		d1 = work(ql, mid);
		d2 = work(mid + 1, qr);
		d = min(d1, d2);
		k = 0;
		for (int i = ql; i <= qr; i++)
		{
			if (abs(point[i].x - point[mid].x) <= d)
				tmp[++k] = i;
		}
		std::sort(tmp + 1, tmp + k + 1, cmpy);
		for (int i = 1; i <= k; i++)
			for (int j = i + 1; j <= min(i + 6, k); j++)
			{
				d = min(d, dist(&point[tmp[i]], &point[tmp[j]]));
			}
		return d;
	}

	void solve() {
		for (int i = 1; i <= n; i++) {
			sum[i] = sum[i - 1] + a[i];
			add_point(1ll * i, sum[i]);
		}
		print(work(1, n));
	}
}

namespace p129 {

	const int MAXN = 1 << 20, MAXM = 1000000;
	int n, a[MAXN];
	int m, q[MAXM];

	void read_case() {
		read(n);
		rep(i, 1, (1 << n)+1)
			read(a[i]);
		read(m);
		rep(i, 0, m)
			read(q[i]);
	}

	LL f[MAXN], g[MAXN];

	void merge_sort(int p, int r, int x)
	{
		if (p >= r)
			return;
		int q = (p + r) / 2;
		merge_sort(p, q, x - 1);
		merge_sort(q + 1, r, x - 1);
		for (int i = p; i <= q; i++)
		{
			f[x] += lower_bound(a + q + 1, a + r + 1, a[i]) - (a + q + 1); //找到右区间中比a[i]小的元素数量,a[i]与它们形成逆序对
			g[x] += (r - (q + 1) + 1) - (upper_bound(a + q + 1, a + r + 1, a[i]) - (a + q + 1)); // 右区间中一共有r-(q+1)+1个元素,找到右区间中小于等于a[i]的元素数量,a[i]与剩下的元素形成顺序对
		}
		sort(a + p, a + r + 1);
	}

	LL flip(int x)
	{
		LL ans = 0;
		for (int i = 1; i <= x; i++)
			swap(f[i], g[i]);
		for (int i = 1; i <= n; i++)
			ans += f[i];
		return ans;
	}

	void solve() {
		int sz = 1 << n;
		merge_sort(1, sz, n);
		for(int i=0; i<m; i++) {
			print(flip(q[i])); // 以2^k分段并翻转
		}
	}
}

namespace p1210 {

	const int INF = 1000000000;
	const int MAX_N = 100000;

	int N;
	LL K;
	int F[MAX_N], W[MAX_N];

	void read_case() {
		read(N, K);
		rep(i, 0, N)
			read(F[i]);
		rep(i, 0, N)
			read(W[i]);
	}

	int jmp[MAX_N][35];
	LL f[MAX_N][35];
	int g[MAX_N][35];

	void build_table()
	{
		for (int i = 0; i < N; i++)
			jmp[i][0] = F[i];

		for (int i = 0; i < N; i++)
			f[i][0] = g[i][0] = W[i];

		for (int j = 1; j <= 34; j++)
		{
			for (int i = 0; i < N; i++)
			{
				jmp[i][j] = jmp[jmp[i][j - 1]][j - 1];
				f[i][j] = f[i][j - 1] + f[jmp[i][j - 1]][j - 1];
				g[i][j] = min(g[i][j - 1], g[jmp[i][j - 1]][j - 1]);
			}
		}
	}

	void query(int u, LL &total, int &minv)
	{
		for (int i = 34; i >= 0; i--)
		{
			if ((K >> i) & 1)
			{
				total += f[u][i];
				minv = min(minv, g[u][i]);
				u = jmp[u][i];
			}
		}
	}

	void solve() {
		build_table();
		rep(i, 0, N) {
			LL total = 0;
			int minv = INF;
			query(i, total, minv);
			print(total, minv);
		}
	}
}

namespace p131 {
	const int MAXN = 100009;
	int n, a[MAXN];
	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(a[i]);
	}

	void solve() {
		int ans = 0;
		for (int i = 1; i <= n; i++)
			if (a[i] > a[i - 1])
				ans += (a[i] - a[i - 1]);
		print(ans);
	}
}

namespace p132 {

	// 这是一个高精度*单精度的类，所以构造函数接收一个整数
	struct BIGNUM {

		int buf[5000]; // 倒着存一个数字

		BIGNUM() { memset(buf, 0, sizeof(buf)); }
		BIGNUM(int n) {
			memset(buf, 0, sizeof(buf));
			for (; n; n /= 10)
				buf[++buf[0]] = n % 10;
		}

		void operator=(BIGNUM& b) {
			memcpy(buf, b.buf, sizeof(buf));
		}

		int& operator[](int index) {
			return buf[index];
		}

		BIGNUM operator/(int b)
		{
			BIGNUM q;
			int k = 0;
			for (int i = buf[0]; i >= 1; i--)
			{
				k = k * 10 + buf[i];
				if (k >= b)
				{
					if (!q[0]) q[0] = i;
					q[i] = k / b;
					k %= b;
				}
			}
			return q;
		}

		BIGNUM operator*(int b)
		{
			BIGNUM c;
			int k = 0;
			for (int i = 1; i <= buf[0]; i++)
			{
				c[i] += buf[i] * b + k;
				c[i + 1] += c[i] / 10;
				c[i] %= 10;
			}
			for (c[0] = buf[0]; c[c[0] + 1];)
			{
				c[0]++;
				c[c[0] + 1] += c[c[0]] / 10;
				c[c[0]] %= 10;
			}
			return c;
		}

		bool operator<=(BIGNUM& b) {
			if (buf[0] > b[0])
				return false;
			else if (buf[0] < b[0])
				return true;
			else
			{
				for (int i = buf[0]; i >= 1; i--)
					if (buf[i] > b[i])
						return false;
					else if (buf[i] < b[i])
						return true;
				return true;
			}
		}

		void print() {
			for (int i = buf[0]; i >= 1; i--)
				printf("%d", buf[i]);
		}
	};

	const int MAX_N = 1000;

	int n; // 大臣数量
	int A, B; //国王手中的数字
	int a[MAX_N], b[MAX_N]; // 大臣们手中的数字

	void read_case() {
		read(n);
		read(A,B);
		rep(i, 0, n)
			read(a[i], b[i]);
	}

	bool cmp(const VI& item0, const VI& item1) {
		return (item0[0] * item0[1]) < (item1[0] * item1[1]);
	}

	void solve() {
		vector<VI> seq;
		rep(i, 0, n) {
			VI  v = { a[i],b[i] };
			seq.push_back(v);
		}
		sort(all(seq), cmp);
		BIGNUM ans;
		BIGNUM K(A);
		rep(i, 0, n) {
			BIGNUM c = K / seq[i][1];
			if (ans <= c)
				ans = c;
			K = K * seq[i][0];
		}
		ans.print();
	}
}

namespace p133 {

	struct BIT {
		VI a;
		BIT(int n) { a = VI(n + 1); }
		int sum(int i) { int s = 0; while (i > 0) s += a[i], i -= (i & -i); return s; }
		void add(int i, int v) { while (i < a.size()) a[i] += v, i += (i & -i); }
	};

	const int MAX_N = 100000;
	int n, a[MAX_N], b[MAX_N];
	
	void read_case() {
		read(n);
		rep(i, 0, n) read(a[i]);
		rep(i, 0, n) read(b[i]);
	}

	void solve() {
		VI ord(n);
		VI a1, b1;
		rep(i, 0, n) a1.push_back(a[i]);
		sort(all(a1));
		rep(i, 0, n)
			ord[lower_bound(all(a1), a[i]) - a1.begin()] = i + 1;
		rep(i, 0, n) b1.push_back(b[i]);
		sort(all(b1));
		rep(i, 0, n)
			b[i] = ord[lower_bound(all(b1), b[i]) - b1.begin()];
		BIT bit = BIT(n);
		int ans = 0;
		repd(i, n - 1, -1) {
			ans = (ans + bit.sum(b[i] - 1)) % (100000000 - 3);
			bit.add(b[i], 1);
		}
		print(ans);
	}
}

namespace p134 {
	
	const int MAX_N = 1000;

	int n, m, k;
	int D[MAX_N];
	int t[MAX_N], a[MAX_N], b[MAX_N];

	void read_case() {
		read(n, m, k);
		rep(i, 1, n)
			read(D[i]);
		rep(i, 0, m)
			read(t[i], a[i], b[i]);
	}

	int MAXN;
	int c[MAX_N];
	int last[MAX_N], e[MAX_N];
	void solve() {
		MAXN = n + 1;
		rep(i, 0, m) {
			last[a[i]] = max(last[a[i]], t[i]);
			c[b[i]] += 1;
		}
		rep(i, 1, n)
			e[i + 1] = max(e[i], last[i]) + D[i];
		int ans = 0;
		rep(i, 0, m) {
			int tm = e[b[i]] - t[i];
			ans += tm;
		}
			
		rep(_, 0, k) {
			int max_save = 0;
			int this_i = 0;
			rep(i, 1, n) {
				int save = 0; //计算对D[i]使用加速器能节约的时间
				if (D[i] > 0)
					rep(j, i + 1, MAXN) { //从j开始抵达时间都会减一
					save += c[j];
					if (e[j] <= last[j]) //意味着从j + 1开始抵达时间不会变化
						break;
				}
				if (save > max_save) {
					max_save = save;
					this_i = i;
				}
			}
			ans -= max_save;
			D[this_i] -= 1;
			rep(i, this_i, n)
				e[i + 1] = max(e[i], last[i]) + D[i];
		}
		print(ans);
	}
}

namespace p135 {

	const int MAX_N = 100000, MAX_M = 100000;

	int n,m;
	int A[MAX_N], B[MAX_N];
	int C[MAX_M], D[MAX_M], K[MAX_M];

	void read_case()
	{
		read(n);
		rep(i, 0, n)
			read(A[i], B[i]);
		read(m);
		rep(i, 0, m)
			read(C[i], D[i], K[i]);
	}

	struct v4
	{
		int lo, hi, k, i;
	};
	bool cmp_lo(const v4& l, const v4& r) { return l.lo < r.lo; }

	v4 a[MAX_N];
	v4 b[MAX_M];
	int ans[MAX_N];

	void solve()
	{
		rep(i, 0, n) // 重新组织数据
			a[i] = { A[i], B[i], 0, i };
		rep(i, 0, m)
			b[i] = { C[i], D[i], K[i], i };
		sort(a, a + n, cmp_lo);
		sort(b, b + m, cmp_lo);

		bool found = true;

		set<pair<int, int> > s;
		int j = 0;
		for (int i = 0; i < n; i++)
		{
			while (j < m && b[j].lo <= a[i].lo)
			{
				s.insert(make_pair(b[j].hi, j));
				j += 1;
			}
			set<pair<int, int> >::iterator iter = s.lower_bound(make_pair(a[i].hi, 0)); //找到大于等于a[i].hi的第一个元素 
			if (iter == s.end())
			{
				found = false;
				break;
			}
			int this_j = iter->second;
			ans[a[i].i] = b[this_j].i + 1;
			b[this_j].k -= 1;
			if (b[this_j].k == 0)
				s.erase(iter);
		}

		if (found)
		{
			printf("YES\n");
			for (int i = 0; i < n; i++)
				printf("%d ", ans[i]);
		}
		else
		{
			printf("NO");
		}
	}
}

namespace p136 {

	const int MAXN = 100009;
	int n, m;
	int c[MAXN], w[MAXN];

	void read_case() {
		scanf("%d %d", &n, &m);
		for (int i = 1; i <= n; i++)
			scanf("%d", &c[i]);
		for (int i = 1; i <= n; i++)
			scanf("%d", &w[i]);
	}

	struct cmp
	{
		bool operator()(int i, int j)
		{
			int a = w[i] * (100 - c[i] % 100);
			int b = w[j] * (100 - c[j] % 100);
			if (a <= b)
				return false;
			else
				return true;
		}
	};

	int ans[MAXN][2]; // ans[i][0]:在第i天花费的100元纸币数 ans[i][1]:花费的1元硬币数
	LL sum; // 愤怒值.注意!要使用ll否则会溢出

	void solve() {
		priority_queue<int, std::vector<int>, cmp> q;

		for (int i = 1; i <= n; i++)
		{
			int k = c[i] % 100;
			ans[i][0] = c[i] / 100;
			ans[i][1] = k;
			m -= k;
			if (k > 0)
			{
				q.push(i);
				if (m < 0)
				{
					int day = q.top(); q.pop();
					m += 100;
					ans[day][0]++;
					ans[day][1] = 0;
					sum += w[day] * (100 - c[day] % 100);
				}
			}
		}
		printf("%lld\n", sum);
		for (int i = 1; i <= n; i++)
			printf("%d %d\n", ans[i][0], ans[i][1]);
	}
}

namespace p137 {
	const int MAX_N = 100000, MAX_M = 100000;

	int n, m;
	int a[MAX_N], b[MAX_M], p[MAX_M];

	void read_case() {
		read(n, m);
		rep(i, 0, n) read(a[i]);
		rep(i, 0, m) read(b[i]);
		rep(i, 0, m) read(p[i]);
	}

	
	void solve() {
		vector<VI> l;
		rep(i, 0, m)
			l.push_back({ b[i], p[i] });
		sort(all(l));
		priority_queue<int, vector<int>, greater<int> > Q;
		int count = 0;
		int j = 0;
		repd(i, n - 1, -1) {
			while (j < m && l[j][0] <= a[i]) {
				Q.push(l[j][1]);
				j += 1;
			}
			int cost = 0;
			while (Q.size() > 0) {
				if (cost + Q.top() <= a[i]) {
					cost += Q.top();
					Q.pop();
					count += 1;
				}
				else
					break;
			}
			int rem = a[i] - cost;
			if (Q.size() > 0) {
				int v = Q.top();
				v -= rem;
				Q.push(v);
			}
		}
		print(count);
	}
}

namespace p138 {

	typedef long long ll;
	typedef struct IDA // int data arr
	{
		ll p, t, acc_t, min, max;
	}IDA, * IDA_PTR;

	const int MAXN = 150009;
	int n;
	ll T;
	IDA ida[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%lld", &ida[i].p);
		for (int i = 1; i <= n; i++)
		{
			scanf("%lld", &ida[i].t);
			T += ida[i].t;
		}
	}

	int cmp1(IDA_PTR i, IDA_PTR j)
	{
		ll a = 1LL * i->t * j->p;
		ll b = 1LL * i->p * j->t;
		if (a < b)
			return(-1);
		else if (a == b)
			return(0);
		else
			return(1);
	}

	int cmp2(IDA_PTR i, IDA_PTR j)
	{
		if (i->p < j->p)
			return(-1);
		else if (i->p == j->p)
			return(0);
		else
			return(1);
	}

	int check(double mid)
	{
		double max_score = -1e18, _max_score = 1e18;
		for (int i = 1; i <= n; i++)
		{
			if (ida[i].p != ida[i - 1].p) _max_score = max_score;
			double score = ida[i].p * (1.0 - mid * ida[i].max / T);
			if (score < _max_score) // 最晚做得分最低,如果不允许这道题最晚做说明给定的C不可行
				return(0);
			score = ida[i].p * (1.0 - mid * ida[i].min / T);
			printf("%f\n", score);
			max_score = max(max_score, score); // 更新最高得分
		}
		return(1);
	}

	void solve() {
		qsort(ida + 1, n, sizeof(IDA), (int (*)(const void*, const void*))cmp1); // 根据 t[i]/p[i]排序得到的做题顺序是最优的
		for (int i = 1; i <= n; i++) // 统计前缀时间,用于计算最终得分
			ida[i].acc_t = ida[i - 1].acc_t + ida[i].t;
		for (int i = 1, j; i <= n; i = j)
		{
			for (j = i; j <= n && cmp1(&ida[i], &ida[j]) == 0; j++)
				;
			for (int k = i; k < j; k++) // 对于ti/pi相同的几道题,在这段时间内不论做题顺序如何最终得分之和是一样的
			{
				ida[k].min = ida[i - 1].acc_t + ida[k].t;
				ida[k].max = ida[j - 1].acc_t;
			}
		}
		qsort(ida + 1, n, sizeof(IDA), (int (*)(const void*, const void*))cmp2); // 再根据p排序
		check(0.328125);
	}
}

}