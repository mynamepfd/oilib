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

namespace p141 {

	const int MAXN = 65;
	int n, a[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &a[i]);
	}

	int done;
	int used[MAXN];
	int total_len, max_len, len;

	void dfs(int tot, int x, int sum)
	{
		if (tot == (total_len / len) - 1) // 剪枝
		{
			done = 1;
			return;
		}
		for (int i = x; i <= n; i++)
		{
			if (!used[i])
			{
				used[i] = 1;
				if (sum + a[i] == len)
				{
					dfs(tot + 1, 1, 0);
					if (done) return;
				}
				else if (sum + a[i] < len)
				{
					dfs(tot, i + 1, sum + a[i]);
					if (done) return;
				}
				used[i] = 0;
				if (sum == 0) // 剪枝
					break;
				while (a[i + 1] == a[i]) // 说明挑选这根木根再加上前面挑选的木棍并继续挑选拼不出一条边，所以跳过后面跟这条木根长度相同的
					i++;
			}
		}
	}

	bool cmp(int x, int y) { return x > y; }

	void solve()
	{
		sort(a + 1, a + n + 1, cmp); // 从大到小排序

		total_len = 0;
		max_len = -1;
		for (int i = 1; i <= n; i++)
		{
			total_len += a[i];
			max_len = max(max_len, a[i]);
		}

		for (len = max_len; len <= total_len; len++) // 
		{
			if (total_len % len != 0)
				continue;

			done = 0;
			memset(used, 0, sizeof(used));

			dfs(0, 1, 0);
			if (done)
			{
				printf("%d\n", len);
				break;
			}
		}
	}	
}

namespace p142 {

	const int MAX_N = 23;

	int n;
	int A[MAX_N], B[MAX_N];

	void read_case() {
		rep(i, 0, n)
			read(A[i], B[i]);
	}

	void init();
	void add_card(int num, int color);
	int find();

	void solve() {
		init();
		rep(i, 0, n)
			add_card(A[i], B[i]);
		print(find());
	}

	typedef struct STATE
	{
		int count[MAX_N];
	}STATE;
	STATE init_state;

	void init()
	{
		memset(&init_state, 0, sizeof(STATE));
	}

	void add_card(int num, int color)
	{
		if (num == 1) // A
			init_state.count[14]++;
		else
			init_state.count[num]++;
	}

	int max_step, ans;
	STATE ans_state;
	vector<STATE> ans_steps;
	void dfs(int cur_step, STATE state, vector<STATE> steps);

	int find()
	{
		max_step = min(n, 13); // 因为最多有13种牌型,所以即便每次只出掉一种牌型,最多13次就可以出光
		ans = 13;
		vector<STATE> steps;
		steps.push_back(init_state);
		dfs(0, init_state, steps);
		//for(int i=0; i<ans_steps.size(); i++)
		//	{
		//	printf("step%d\n",i+1);
		//	print(ans_steps[i]);
		//	}
		return ans;
	}

	int need[4] = {
		0,
		5, // 想出顺子最少需要五种连着的牌型,例如56789.
		3, // 想出连对最少需要三种连着的牌型,例如334455.
		2, // 想出飞机最少需要两种连着的牌型,例如333444.
	};
	int remain(STATE state);

	void dfs(int cur_step, STATE state, vector<STATE> steps)
	{
		if (cur_step >= max_step)
			return;
		for (int i = 3; i >= 1; i--) // 依次检查能否出飞机、连对、顺子
			for (int p = 3; p <= 13; p++)
			{
				int q = p;
				while (q <= 14 && state.count[q] >= i) q++;
				q--; // 找到了从p到q的连着的牌型
				int len = q - p + 1;
				if (len >= need[i]) // 比如找到了3344556677,
				{
					STATE new_state = state;
					for (int k = p; k <= p + need[i] - 2; k++) // 想出连对是必须先出3344
						new_state.count[k] -= i;
					for (int k = p + need[i] - 1; k <= q; k++) // 检查出334455或33445566或33445566771的效果
					{
						new_state.count[k] -= i;
						steps.push_back(new_state);
						dfs(cur_step + 1, new_state, steps);
					}
				}
			}
		int val = cur_step + remain(state);
		if (val < ans)
		{
			ans = val;
			ans_state = state;
			ans_steps = steps;
		}
	}

	int remain(STATE state)
	{
		int b[MAX_N], res = 0;
		memset(b, 0, sizeof(b));
		for (int i = 0; i <= 14; i++) // 统计剩下的牌中牌型的数量,比如AAAA 7788
		{
			if (i == 1)
				continue;
			b[state.count[i]]++; // 统计之后得b[4]=1,b[2]=2
		}

		while (b[4] >= 1 && b[2] >= 2) { b[4]--; b[2] -= 2; res++; }// 四带二对
		while (b[4] >= 1 && b[1] >= 2) { b[4]--; b[1] -= 2; res++; }// 四带二
		while (b[3] >= 1 && b[2] >= 1) { b[3]--; b[2] -= 1; res++; }// 三带对
		while (b[3] >= 1 && b[1] >= 1) { b[3]--; b[1] -= 1; res++; }// 三带一

		res += b[4] + b[3] + b[2] + b[1]; // 出单牌
		return res;
	}	
}

namespace p143 {

	int n, a[5][7]; // 原点在左下角,x轴向右,y轴向上

	void read_case() {
		scanf("%d", &n);
		for (int x = 0; x < 5; x++)
		{
			for (int y = 0;; y++)
			{
				int c;
				scanf("%d", &c);
				if (c == 0)
					break;
				a[x][y] = c;
			}
		}
	}

	struct VECTOR3D { int x, y, z; };
	VECTOR3D vec3d(int x, int y, int z) { VECTOR3D v; v.x = x; v.y = y; v.z = z; return v; }
	typedef std::vector<VECTOR3D> STATE;

	int done;
	STATE ans;
	void dfs(int dep, STATE state);

	void solve() {
		STATE s;
		dfs(0, s);

		if (!done)
			printf("-1");
		else
		{
			for (int i = 0; i < ans.size(); i++)
			{
				VECTOR3D& v = ans[i];
				printf("%d %d %d\n", v.x, v.y, v.z);
			}
		}
	}

	int check();
	int fall();
	int erase();

	void dfs(int tot, STATE s)
	{
		if (tot >= n)
		{
			if (check())
			{
				done = 1;
				ans = s;
			}
			return;
		}
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
			{
				if (a[x][y] == 0)
					continue;
				if (x < 4) // 与右边的交换
				{
					int a_[5][7];
					memcpy(a_, a, sizeof(a));

					swap(a[x][y], a[x + 1][y]);
					fall();
					while (erase())
						fall();

					STATE s1 = s;
					s1.push_back(vec3d(x, y, 1));
					dfs(tot + 1, s1);

					memcpy(a, a_, sizeof(a));
					if (done) return;
				}
				if (x && !a[x - 1][y]) // 当左边为空时与左边交换
				{
					int a_[5][7];
					memcpy(a_, a, sizeof(a));

					swap(a[x][y], a[x - 1][y]);
					fall();
					while (erase())
						fall();

					STATE s1 = s;
					s1.push_back(vec3d(x, y, -1));
					dfs(tot + 1, s1);

					memcpy(a, a_, sizeof(a));
					if (done) return;
				}
			}
	}

	int check()
	{
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 7; j++)
				if (a[i][j])
					return(0);
		return(1);
	}

	int fall() // 如果产生了下落则返回(1)否则返回(0)
	{
		int flag;
		flag = 0;
		for (int x = 0; x < 5; x++)
			for (int y = 1; y < 7; y++)
			{
				if (!a[x][y] || a[x][y - 1])
					continue; // 如果(x,y)是空的或者(x,y)已经落在(x,y-1)上都不需要下落
				flag = 1;
				int y1;
				for (y1 = y - 1; y1 >= 0 && !a[x][y1]; y1--)
					; // y1一直往下直到找到第一个非空的格子
				a[x][y1 + 1] = a[x][y];
				a[x][y] = 0;
			}
		return flag;
	}

	int erase()
	{
		int flag;
		int to_clear[5][7];

		flag = 0;
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
				to_clear[x][y] = 0;

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 7; y++)
			{
				if (!a[x][y])
					continue;
				int x1;
				for (x1 = x; x1 < 5 && a[x1][y] == a[x][y]; x1++)
					; // x1一直往右直到找到第一个跟a[x][y]不同的格子
				if (x1 - x >= 3) // 连续三个以上可以消除
				{
					flag = 1;
					for (int x1 = x; x1 < 5 && a[x1][y] == a[x][y]; x1++)
						to_clear[x1][y] = 1;

				}
			}
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 5; y++)
			{
				if (!a[x][y])
					continue;
				int y1;
				for (y1 = y; y1 < 7 && a[x][y1] == a[x][y]; y1++)
					; // y1一直往上直到找到第一个跟a[x][y]不同的格子
				if (y1 - y >= 3)
				{
					flag = 1;
					for (y1 = y; y1 < 7 && a[x][y1] == a[x][y]; y1++)
						to_clear[x][y1] = 1;
				}
			}

		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
			{
				if (to_clear[x][y])
					a[x][y] = 0;
			}

		return flag;
	}
}

namespace p144 {
	
	const int MAXW = 20;
	const int MAXH = 20;

	const int BLOCK = 1;
	const int S = 2;
	const int G = 3;

	int w, h;
	int a[MAXW][MAXH];
	int sx, sy;

	void read_case() {
		scanf("%d %d", &w, &h);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				scanf("%d", &a[x][y]);
				if (a[x][y] == S)
				{
					sx = x; sy = y;
				}
			}
	}

	struct VECTOR3D { int x, y, z; };
	VECTOR3D vec3d(int x, int y, int z) { VECTOR3D v; v.x = x; v.y = y; v.z = z; return v; }
	typedef std::vector<VECTOR3D> STATE;

	const int INF = 1000000000;
	int ans;
	void dfs(int tot, int x, int y, STATE s);

	void solve()
	{
		ans = INF;

		STATE s;
		dfs(0, sx, sy, s);

		if (ans != INF)
			printf("%d\n", ans);
		else
			printf("-1\n");
	}

	int dir[4][2] =
	{
	{0,-1},
	{1,0},
	{0,1},
	{-1,0}
	};

	void dfs(int tot, int x, int y, STATE s)
	{
		if (tot > min(10, ans))
			return;

		for (int d = 0; d < 4; d++)
		{
			int nx = x + dir[d][0], ny = y + dir[d][1];
			if (nx<0 || nx>w - 1 || ny<0 || ny>h - 1)
				continue;
			if (a[nx][ny] == BLOCK)
				continue;
			int cx, cy;
			for (;;)
			{
				cx = nx; cy = ny;
				if (a[cx][cy] == G)
				{
					if (tot + 1 < ans && tot + 1 <= 10)
						ans = tot + 1;
					break;
				}
				nx = cx + dir[d][0]; ny = cy + dir[d][1];
				if (nx<0 || nx>w - 1 || ny<0 || ny>h - 1)
				{
					break;
				}
				if (a[nx][ny] == BLOCK)
				{
					a[nx][ny] = 0;
					STATE s1 = s;
					//s1.push_back(vec3d(x,y,d));
					dfs(tot + 1, cx, cy, s1);
					a[nx][ny] = BLOCK;
					break;
				}
			}
		}
	}
}

namespace p145 {
	
	const int MAXM = 29;
	int N, m;

	void read_case() {
		scanf("%d%d", &N, &m);
	}

	struct VECTOR2D { int x, y; };
	VECTOR2D vec2d(int x, int y) { VECTOR2D v; v.x = x; v.y = y; return v; }
	typedef std::vector<VECTOR2D> STATE;

	const int INF = 1000000000;
	int ans;
	int low_s[MAXM], low_v[MAXM];
	void dfs(int dep, int S, int V, int max_r, int max_h, STATE state);

	void solve()
	{
		// 计算第1开始到第m层的最小表面积和体积
		for (int i = 1; i <= m; i++)
		{
			low_s[i] = low_s[i - 1] + 2 * i * i;
			low_v[i] = low_v[i - 1] + i * i * i;
		}

		ans = INF;

		STATE s;
		dfs(m, 0, 0, 100, 10000, s);
		if (ans == INF)
			printf("0\n");
		else
			printf("%d\n", ans);
	}

	void dfs(int dep, int S, int V, int max_r, int max_h, STATE s)
	{
		if (dep == 0)
		{
			if (V == N && S < ans)
			{
				ans = S;
			}
			return;
		}
		if (V > N || S > ans) // 剪枝
			return;

		if ((V + low_v[dep]) > N || (S + low_s[dep] > ans)) // 如果剩下的层都用最小体积的蛋糕...
			return;

		if ((2 * (N - V) / max_r + S) > ans) // 重要!如果把剩下的体积做成一个蛋糕所得的表面积
			return;

		for (int r = max_r - 1; r >= dep; r--)
		{
			if (dep == m) S = r * r;
			for (int h = max_h - 1; h >= dep; h--)
			{
				//STATE s1 = s;
				//s1.push_back(vec2d(r,h));
				dfs(dep - 1, S + 2 * r * h, V + r * r * h, r, h, s);
			}
		}
	}
}

namespace p146 {

	const int MAXX = 6;
	const int MAXY = 6;
	int a[6][6];

	void read_case() {
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				scanf("%d", &a[x][y]);
	}

	void init();
	int bfs();

	void solve() {
		init();
		int ans = bfs();
		if (ans > 20)
			printf("too difficult\n");
		else
			printf("%d\n", ans);
	}

	typedef long long ll;
	typedef struct STATE
	{
		int tot;
		int a[6][6], x, y;
	}STATE, * STATE_PTR;

	STATE st, ed;
	std::queue<STATE> q;
	std::unordered_map<ll, int> M, N;
	int ans;
	STATE final_state;

	void init()
	{
		while (!q.empty())
			q.pop();
		M.clear();
		N.clear();
		ans = 21;
	}

	int dir[4][2] =
	{
	{-1,-1},
	{0,-1},
	{0,1},
	{1,1}
	};
	ll hash(STATE_PTR state);

	int bfs()
	{
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				st.a[x][y] = y;
		st.x = st.y = 0;

		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
			{
				ed.a[x][y] = a[x][y];
				if (ed.a[x][y] == 0)
				{
					ed.x = x;
					ed.y = y;
				}
			}

		q.push(st);
		M[hash(&st)] = 0;
		while (!q.empty())
		{
			STATE s = q.front(); q.pop();
			for (int d = 0; d < 4; d++)
			{
				int nx = s.x + dir[d][0], ny = s.y + dir[d][1];
				if (ny >= 0 && ny < 6 && nx >= 0 && nx <= ny) // 第y行的x坐标范围是[0,y]
				{
					STATE s1 = s;
					swap(s1.a[s1.x][s1.y], s1.a[nx][ny]);
					s1.x = nx; s1.y = ny;
					s1.tot++;
					ll ha = hash(&s1);
					if (M.find(ha) == M.end() && s1.tot <= 10)
					{
						M[ha] = s1.tot;
						q.push(s1);
					}
				}
			}
		}

		ll ha = hash(&ed);
		N[ha] = 0;
		q.push(ed);
		if (M.find(ha) != M.end())
		{
			ans = M[ha];
			//final_state = ed;
			return ans;
		}

		while (!q.empty())
		{
			STATE s = q.front(); q.pop();
			for (int d = 0; d < 4; d++)
			{
				int nx = s.x + dir[d][0], ny = s.y + dir[d][1];
				if (ny >= 0 && ny < 6 && nx >= 0 && nx <= ny) // 第y行的x坐标范围是[0,y]
				{
					STATE s1 = s;
					swap(s1.a[s1.x][s1.y], s1.a[nx][ny]);
					s1.x = nx; s1.y = ny;
					s1.tot++;
					ll ha = hash(&s1);
					if (N.find(ha) == N.end() && s1.tot <= 10)
					{
						N[ha] = s1.tot;
						q.push(s1);
						if (M.find(ha) != M.end())
						{
							ans = min(ans, s1.tot + M[ha]);
							//final_state = s1;
						}
					}
				}
			}
		}

		return ans;
	}

	ll hash(STATE_PTR state)
	{
		ll val = 0;
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				val = 6LL * val + state->a[x][y];
		return val;
	}
}

namespace p151 {

	const int MAXN = 300000;
	int n, m, q, x[MAXN + 1], y[MAXN + 1];

	void read_case() {
		read(n, m, q);
		rep(i, 1, q + 1)
			read(x[i], y[i]);
	}

	typedef long long ll;

	class Interval // 鍔ㄦ�佸紑鐐?
	{
	public:
		int l, r, sum;
		Interval* lc, * rc;
		Interval() {}
		Interval(int l, int r)
		{
			this->l = l; this->r = r;
			lc = rc = NULL;
			sum = 0;
		}
		void update()
		{
			sum = 0;
			if (lc != NULL)
				sum += lc->sum;
			if (rc != NULL)
				sum += rc->sum;
		}
		void change(int p, int v)
		{
			if (l == r)
			{
				sum = v;
				return;
			}
			int mid = (l + r) / 2;
			if (lc == NULL)
				lc = new Interval(l, mid);
			if (p <= lc->r)
				lc->change(p, v);
			else
			{
				if (rc == NULL)
					rc = new Interval(mid + 1, r);
				rc->change(p, v);
			}

			update();
		}
		int query(int k)
		{
			if (l == r)return l;
			int mid = (l + r) / 2;
			if (lc == NULL)
				lc = new Interval(l, mid);

			int tot = mid - l + 1 - lc->sum;
			if (k <= tot)
				return lc->query(k);
			else
			{
				if (rc == NULL)
					rc = new Interval(mid + 1, r);
				return rc->query(k - tot);
			}
		}
	};

	Interval* row[MAXN + 1], * col;
	vector<ll> a[MAXN + 1], b;

	void solve() {
		rep(i, 1, n + 1)
		{
			row[i] = new Interval(1, m + q);
		}
		col = new Interval(1, n + q);
		rep(i, 1, q + 1)
		{
			if (y[i] == m)
			{
				int j = col->query(x[i]);
				ll id;
				if (j <= n)
					id = 1LL * j * m;
				else
					id = b[j - n - 1];
				col->change(j, 1);
				col->change(n + i + 1, 0);
				b.push_back(id);
				print(id);
				print();
			}
			else
			{
				int j = row[x[i]]->query(y[i]);
				ll id1, id2;
				if (j <= m - 1)
					id1 = 1LL * (x[i] - 1) * m + j;
				else
					id1 = a[x[i]][j - (m - 1) - 1];
				row[x[i]]->change(j, 1);
				row[x[i]]->change((m - 1) + i + 1, 0);
				j = col->query(x[i]);
				if (j <= n)
					id2 = 1LL * j * m;
				else
					id2 = b[j - n - 1];
				col->change(j, 1);
				col->change(n + i + 1, 0);
				a[x[i]].push_back(id2);
				b.push_back(id1);
				print(id1);
				print();
			}
		}
	}
}

namespace p152_part1 {

	const int MAXN = 100009;
	int n, a[MAXN], b[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &a[i]);
		for (int i = 1; i <= n; i++)
			scanf("%d", &b[i]);
	}

	typedef struct VECTOR3
	{
		int x, y, z;
		bool operator<(const VECTOR3& other) const
		{
			return other.z < z;
		}
	}VECTOR3;
	priority_queue<VECTOR3> q;

	void solve() {
		for (int i = 1; i <= n; i++)
		{
			VECTOR3 vec;
			vec.x = i;
			vec.y = 1;
			vec.z = a[i] + b[1];
			q.push(vec);
		}

		for (int k = 1; k <= n; k++)
		{
			VECTOR3 vec = q.top(); q.pop();
			int x, y, z;
			printf("%d ", vec.z);
			vec.y++;
			vec.z = a[vec.x] + b[vec.y];
			q.push(vec);
		}
	}
}

namespace p152_part2 {

	void read_case() {

	}

	typedef long long ll;
	void init();
	void next(ll* p);

	void solve() {
		ll ans;

		init();
		for (int i = 1; i <= 1500; i++)
			next(&ans);
		printf("The 1500'th ugly number is %lld.\n", ans);
	}

	struct cmp
	{
		bool operator()(ll i, ll j)
		{
			return i > j;
		}
	};
	std::priority_queue<ll, std::vector<ll>, cmp> q;

	void init()
	{
		q.push(1LL);
	}

	void next(ll* num)
	{
		*num = q.top();
		q.push(*num * 2LL);
		q.push(*num * 3LL);
		q.push(*num * 5LL);
		while (q.top() == *num)
			q.pop();
	}
}

namespace p162 {

	const int MAXN = 100001;
	int n, h[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &h[i]);
	}

	struct
	{
		//STATE state;
		int len;
	} f[MAXN][2]; // f[i][0]记录点i为波谷的最长序列 f[i][1]记录点i为波峰的最长序列

	int ans;

	void solve()
	{
		/*f[1][0].state.push_back(h[1]);*/ f[1][0].len = 1;
		/*f[1][1].state.push_back(h[1]);*/ f[1][1].len = 1;
		for (int i = 2; i <= n; i++)
		{
			for (int j = max(1, i - 1000); j < i; j++)
			{
				if (h[j] < h[i] && f[j][1].len + 1 > f[i][0].len) // 形成波谷
				{
					f[i][0] = f[j][1];
					//f[i][0].state.push_back(h[i]);
					f[i][0].len++;
				}
				if (h[j] > h[i] && f[j][0].len + 1 > f[i][1].len)
				{
					f[i][1] = f[j][0];
					//f[i][1].state.push_back(h[i]);
					f[i][1].len++;
				}
			}
		}



		int ans = -1;
		for (int i = 1; i <= n; i++)
			ans = max(ans, max(f[i][0].len, f[i][1].len));

		printf("%d\n", ans);
	}
}

namespace p163 {

	const int  MAXN = 2509;
	const int MAX_Q = 100000;
	const int  PLACE =1;
	const int  REMOVE =2;
	const int  WALK =3;

	int n, m, q;
	int op[MAX_Q], r1[MAX_Q], c1[MAX_Q], r2[MAX_Q], c2[MAX_Q];

	void read_case() {
		scanf("%d%d%d", &n, &m, &q);
		rep(i,0,q)
			scanf("%d%d%d%d%d", &op[i], &r1[i], &c1[i], &r2[i], &c2[i]);
	}

	void add_fence(int r1, int c1, int r2, int c2);
	void remove_fence(int r1, int c1, int r2, int c2);
	int walk(int r1, int c1, int r2, int c2);

	void solve() {
		rep(i, 0, q) {
			switch (op[i]) {
			case PLACE:
				add_fence(r1[i], c1[i], r2[i], c2[i]);
				break;
			case REMOVE:
				remove_fence(r1[i], c1[i], r2[i], c2[i]);
				break;
			case WALK:
				if (walk(r1[i], c1[i], r2[i], c2[i]))
					printf("Yes\n");
				else
					printf("No\n");
				break;
			default:
				break;
			}
		}
		
	}

	typedef long long ll;
	typedef struct FENCE
	{
		int r1, c1, r2, c2;
	}FENCE, * FENCE_PTR;
	unordered_map<ll, int> M;

	mt19937 random;
	ll hash(FENCE_PTR fence);
	void add(int r, int c, int v);
	int sum(int r, int c);

	void add_fence(int r1, int c1, int r2, int c2)
	{
		int v = random();

		FENCE fence;
		fence.r1 = r1;
		fence.c1 = c1;
		fence.r2 = r2;
		fence.c2 = c2;
		M[hash(&fence)] = v;

		add(r1, c1, v);
		add(r1, c2 + 1, -v);
		add(r2 + 1, c1, -v);
		add(r2 + 1, c2 + 1, v);
	}

	void remove_fence(int r1, int c1, int r2, int c2)
	{
		FENCE fence;
		fence.r1 = r1;
		fence.c1 = c1;
		fence.r2 = r2;
		fence.c2 = c2;
		int v = M[hash(&fence)];
		add(r1, c1, -v);
		add(r1, c2 + 1, v);
		add(r2 + 1, c1, v);
		add(r2 + 1, c2 + 1, -v); // 注意!当r2=N,c2=M时,再加1就会越界,所以数组要开大点
	}

	int walk(int r1, int c1, int r2, int c2)
	{
		int v1 = sum(r1, c1);
		int v2 = sum(r2, c2);
		return v1 == v2;
	}

	int bit[MAXN][MAXN];
	void add(int r, int c, int v)
	{
		for (int i = r; i <= n; i += i & -i)
			for (int j = c; j <= m; j += j & -j)
				bit[i][j] += v;
	}

	int sum(int r, int c)
	{
		int res = 0;
		for (int i = r; i >= 1; i -= i & -i)
			for (int j = c; j >= 1; j -= j & -j)
				res += bit[i][j];
		return res;
	}

	ll hash(FENCE_PTR fence)
	{
		return fence->r1 * 15643757501LL + fence->c1 * 6255001LL + fence->r2 * 2501 + fence->c2;
	}
}

namespace p171 {

	const int MAXN = 200000;
	const int MAX_Q = 50000;
	int n, q, L[MAX_Q], R[MAX_Q];

	void read_case() {
		read(n, q);
		rep(i, 0, q)
			read(L[i], R[i]);
	}

	const int MAXB = 450;
	int a[MAXN + 1], btot, bl[MAXB], br[MAXB], id[MAXN + 1];
	VI b[MAXB];
	void divide();
	LL query(int l, int r, int v);

	void solve() {
		rep(i, 1, n + 1)
			a[i] = i;
		divide();
		LL ans = 0;
		rep(i, 0, q)
		{
			int l = L[i];
			int r = R[i];
			if (l == r)
			{
				print(ans);
				continue;
			}
			if (l > r) // NOTE!!!
				swap(l, r);

			LL cnt1 = query(l + 1, r - 1, a[l]);
			LL cnt2 = query(l + 1, r - 1, a[r]);
			ans += 2 * (cnt1 - cnt2);
			if (a[l] > a[r])
				ans--;
			else // a[l]<a[r]
				ans++;
			print(ans);

			b[id[l]].erase(lower_bound(all(b[id[l]]), a[l]));
			b[id[l]].insert(upper_bound(all(b[id[l]]), a[r]), a[r]);
			b[id[r]].erase(lower_bound(all(b[id[r]]), a[r]));
			b[id[r]].insert(upper_bound(all(b[id[r]]), a[l]), a[l]);
			swap(a[l], a[r]);

		}
	}

	void divide()
	{
		int sz = int(ceil(sqrt(n)));
		btot = n / sz + (n % sz > 0);
		rep(i, 1, btot + 1)
		{
			bl[i] = (i - 1) * sz + 1;
			br[i] = min(n, i * sz);
			rep(j, bl[i], br[i] + 1)
			{
				id[j] = i;
				b[i].push_back(j);
			}
		}
	}

	// find the number of integers which greater than s
	LL query(int l, int r, int v)
	{
		if (l > r)
			return 0;
		LL cnt = 0;
		if (id[l] == id[r]) {
			rep(i, l, r + 1)
				if (a[i] > v)
					cnt++;
		}
		else {
			rep(i, l, br[id[l]] + 1) {
				if (a[i] > v)
					cnt++;
			}
			rep(i, id[l] + 1, id[r]) {
				cnt += b[i].end() - upper_bound(all(b[i]), v);
			}
			rep(i, bl[id[r]], r + 1) {
				if (a[i] > v)
					cnt++;
			}
		}

		return cnt;
	}
}

namespace p172 {

	const int MAX_N = 100000, MAX_M = 100000;
	int n, c, m; // 读取文章字数、汉字种类数、询问数
	int a[MAX_N], L[MAX_M], R[MAX_M];

	void read_case() {
		read(n, c, m);
		rep(i, 0, n)
			read(a[i]);
		rep(i, 0, m)
			read(L[i], R[i]);
	}

	struct SqrtDecomp {
		VI a;
		vector<VI> h;
		int n, m, sz; // 数列长度,块数量,块大小

		vector<VI> f;
		SqrtDecomp(VI& a_, int c_, int M) { // c_表示a_中数字的范围
			a = a_;
			n = a.size();
			//sz = pow(n,1.0/3); // 让块变小,减少二分的次数
			sz = max(1, (int)(n / sqrt(M * log2(n)))); // 最优值
			m = n / sz + 1;

			h = vector<VI>(c_ + 1);
			rep(i, 0, a.size()) {
				h[a[i]].push_back(i);
			}

			f = vector<VI>(m, VI(m));
			rep(i, 0, m) {
				VI cnt(c_ + 1);
				int ans = 0; // 出现偶数次的字符的数量
				rep(j, i, m) {
					//print(i,j);
					rep(k, j * sz, min((j + 1) * sz, n)) { //注意这里是j
						if (cnt[a[k]] > 0 && cnt[a[k]] % 2 == 0)
							ans -= 1; // 偶变奇
						else if (cnt[a[k]] % 2 == 1)
							ans += 1; // 奇变偶
						cnt[a[k]]++;
					}
					f[i][j] = ans;
				}
			}

			mp = VI(c_ + 1);
		}

		// 打印第i块到第j块的元素,用于调试
		//void print(int lb, int rb) {
		//	VI t;
		//	rep(i, lb * sz, min((rb + 1) * sz, n))
		//		t.push_back(a[i]);
		//	print(t);
		//}

		VI q, mp;
		int query(int l, int r) {
			int lb = l / sz;
			int rb = r / sz;
			if (lb == rb || lb + 1 == rb) {
				/* TODO (#1#): 当l,r在同一块或相邻块 */
				q.clear();
				rep(i, l, r)
					if (mp[a[i]]++ == 0) q.push_back(a[i]);
				int ans = 0;
				rep(i, 0, q.size()) {
					if (mp[q[i]] % 2 == 0)ans++;
					mp[q[i]] = 0;
				}
				return ans;
			}
			else {
				q.clear();
				rep(i, l, (lb + 1) * sz) {
					if (mp[a[i]]++ == 0) q.push_back(a[i]);
				}
				rep(i, rb * sz, r) {
					if (mp[a[i]]++ == 0) q.push_back(a[i]);
				}

				int ans = f[lb + 1][rb - 1];
				//print(lb+1,rb-1);

				rep(i, 0, q.size()) {
					int t = upper_bound(all(h[q[i]]), rb * sz - 1) -
						lower_bound(all(h[q[i]]), (lb + 1) * sz);
					if (t == 0) {
						if (mp[q[i]] % 2 == 0)
							ans += 1;
					}
					else if (t % 2 == 1) {
						if (mp[q[i]] % 2 == 1)
							ans += 1;
					}
					else {
						if (mp[q[i]] % 2 == 1)
							ans -= 1;
					}
					mp[q[i]] = 0;
				}
				return ans;
			}
		}
	};

	void solve() {
		VI t;
		rep(i, 0, n)
			t.push_back(a[i]);

		SqrtDecomp sd(t, c, m); // 平方分割

		int ans = 0;
		rep(i,0,m) {
			int l, r, tl, tr; 
			l = L[i];
			r = R[i];
			tl = (l + ans) % n;
			tr = (r + ans) % n;
			if (tl > tr)
				swap(tl, tr);
			ans = sd.query(tl, tr + 1);
			print(ans);
		}
	}
}

}

