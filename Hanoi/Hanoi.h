#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
class Hanoi {
private:
	int _num_poles;
	int _num_discs;
	vector<vector<vector<string>>> _cache;
	std::string lookup_moves(int num_discs, int src, int dst);
	std::string get_moves(int num_discs, int src, int dst, int tmp);
public:
	std::string solve(int num_discs, int src, int dst, int tmp);
	//void display();
	friend class Tests;
};