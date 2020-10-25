#include "Hanoi.h"
//Solves the Tower of Hanoi problem recursively and caches the result for a quicker subsequent response
string Hanoi::get_moves(int num_discs, int src, int dst, int tmp) {
    string sequence = "";
    if (num_discs == 0) { // Base case
        return sequence;
    }
    if (lookup_moves(num_discs, src, dst) != "") {
        return sequence += _cache[num_discs][src][dst];
    }
    else {
        sequence += get_moves(num_discs - 1, src, tmp, dst);
        sequence += to_string(src) + "->" + to_string(dst) + "\n";
        sequence += get_moves(num_discs - 1, tmp, dst, src);
    }
    // Resize the sub-caches just enough to fit what you're trying to store
    // clear everything beneath.
    _cache[num_discs].resize(src + 1);
	_cache[num_discs][src].resize(dst + 1);
	_cache[num_discs][src][dst] = sequence;
	_cache[num_discs - 1].clear();
	return sequence;
}
string Hanoi::lookup_moves (int num_discs, int src, int dst) {
    size_t discCopy = num_discs;
    if ((_cache.empty()) || (_cache.size() < discCopy + 1)) {
        _cache.resize(discCopy + 1);
    }
    else if (_cache.size() > num_discs) {
        if (_cache[num_discs].size() > src) {
            if (_cache[num_discs][src].size() > dst) {
                return _cache[num_discs][src][dst];
            }
        }
    }
    return "";
}
string Hanoi::solve(int num_discs, int src, int dst, int tmp) {
    string preface = "# Below, 'A->B' means 'move the top disc on pole A to pole B'\n";
    return preface + get_moves(num_discs, src, dst, tmp);
}
//void Hanoi::display() {
//    for (size_t i = 0; i < _cache.size(); i++) {
//        for (size_t j = 0; j < _cache[i].size(); j++) {
//            for (size_t k = 0; k < _cache[i][j].size(); k++) {
//                cout << i << j << k << endl;
//                cout << _cache[i][j][k] << endl;
//            }
//        }
//    }
//}
//int main()
//{
//    Hanoi test;
//    cout << test.solve(15, 1, 2, 3);
//    cout << test.solve(15, 1, 2, 3);
//    test.display();
//}