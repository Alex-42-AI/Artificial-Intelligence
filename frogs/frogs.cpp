#include<iostream>
#include<vector>
#include<chrono>

using namespace std;
using namespace std::chrono;

int n, mismatches;
const char symbols[3] = { '_', '<', '>' };

void print_out(vector<int>& position, vector<int>& st) {
    for (int i : position)
        cout << symbols[i];

    cout << endl;
    int last = n;

    for (int index : st) {
        swap(position[last], position[index]);

        for (int i : position)
            cout << symbols[i];

        cout << endl;

        last = index;
    }
}

void dfs(int *position, vector<int>& st, int index) {
    if (index && position[index - 1] == 2) {
        position[index] = 2;
        position[index - 1] = 0;
        st.push_back(index - 1);
        int off = n == index - 1 ? -2 : index == n;
        mismatches += off;
        dfs(position, st, index - 1);

        if (!mismatches)
            return;

        position[index - 1] = 2;
        position[index] = 0;
        st.pop_back();
        mismatches -= off;
    }

    if (index > 1 && position[index - 2] == 2 && position[index - 1] == 1) {
        position[index] = 2;
        position[index - 2] = 0;
        st.push_back(index - 2);
        int off = index == n;

        if (n == index - 1)
            off = -1;

        else if (n == index - 2)
            off = -2;

        mismatches += off;
        dfs(position, st, index - 2);

        if (!mismatches)
            return;

        position[index - 2] = 2;
        position[index] = 0;
        st.pop_back();
        mismatches -= off;
    }

    if (index < 2 * n && position[index + 1] == 1) {
        position[index] = 1;
        position[index + 1] = 0;
        st.push_back(index + 1);
        int off = n == index + 1 ? -2 : index == n;
        mismatches += off;
        dfs(position, st, index + 1);

        if (!mismatches)
            return;

        position[index + 1] = 1;
        position[index] = 0;
        st.pop_back();
        mismatches -= off;
    }

    if (index < 2 * n - 1 && position[index + 2] == 1 && position[index + 1] == 2) {
        position[index] = 1;
        position[index + 2] = 0;
        st.push_back(index + 2);
        int off = index == n;

        if (n == index + 1)
            off = -1;

        else if (n == index + 2)
            off = -2;

        mismatches += off;
        dfs(position, st, index + 2);

        if (!mismatches)
            return;

        position[index + 2] = 1;
        position[index] = 0;
        st.pop_back();
        mismatches -= off;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cin >> n;

    int *start = new int[2 * n + 1];

    for (int i = 0; i < n; i++)
        start[i] = 2;

    start[n] = 0;

    for (int i = 1; i <= n; i++)
        start[n + i] = 1;

    mismatches = 2 * n;
    vector<int> result = {};
    time_point t0 = steady_clock::now();
    dfs(start, result, n);
    time_point t1 = steady_clock::now();
    double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    cout << "# TIMES_MS: alg=" << ms << endl;
    vector<int> original = {};
    original.reserve(2 * n + 1);

    for (int i = 0; i < n; i++)
        original.push_back(2);

    original.push_back(0);

    for (int i = 0; i < n; i++)
        original.push_back(1);

    print_out(original, result);

    delete[] start;

    return 0;
}
