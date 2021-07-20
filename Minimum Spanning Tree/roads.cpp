// Utilizes Bay Area road data and constructs a minimum spanning tree.

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <queue>
#include <set>

using std::cout;
using std::endl;
using std::exception;
using std::ifstream;
using std::map;
using std::sqrt;
using std::string;
using std::vector;
using std::queue;
using std::priority_queue;
using std::pair;
using std::set;
using std::ofstream;

class MyException : public exception {
    const char* msg;
public:
    MyException(const char* msg) : msg(msg) {}

    const char* what() const noexcept {
        return msg;
    }
};

typedef int_least64_t OSMID;

struct Node {
    double x, y;
};

struct Road {
    string name;
    vector<OSMID> path;
};

map<OSMID, Node> load_nodes(const string& file_name) {
    ifstream nodes_file(file_name);
    map<OSMID, Node> nodes;
    double min_x, min_y, max_x, max_y;
    if (nodes_file.bad() || nodes_file.fail()) {
        throw MyException("Error opening nodes file.");
    }
    nodes_file >> min_x >> min_y >> max_x >> max_y;
    while (nodes_file.good()) {
        OSMID osmid;
        double x, y;
        nodes_file >> osmid >> x >> y;
        if (nodes_file.good()) {
            nodes[osmid] = { x, y };
        }
    }
    if (nodes_file.bad()) {
        throw MyException("Error reading nodes from file.");
    }
    nodes_file.close();
    return nodes;
}

map<OSMID, Road> load_roads(const string& file_name) {
    map<OSMID, Road> roads;
    ifstream roads_file(file_name);
    long num_roads = 0;
    if (roads_file.bad() || roads_file.fail()) {
        throw MyException("Error opening roads file.");
    }
    while (roads_file.good()) {
        OSMID osmid;
        int path_len;
        roads_file >> osmid >> path_len;
        if (roads_file.good()) {
            Road r;
            r.path.reserve(path_len);
            for (int j = 0; j < path_len; ++j) {
                OSMID node_id;
                roads_file >> node_id;
                ++num_roads;
                r.path.push_back(node_id);
            }
            roads_file.get(); // Read the extra space
            getline(roads_file, r.name);
            roads[osmid] = r;
        }
    }
    if (roads_file.bad()) {
        throw MyException("Error reading roads from file.");
    }
    roads_file.close();
    return roads;
}

double distance(const Node& a, const Node& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double get_road_length(const map<OSMID, Node>& nodes, const Road& road) {
    double len = 0.0;
    for (size_t i = 1; i < road.path.size(); ++i) {
        const Node& a = nodes.find(road.path[i - 1])->second;
        const Node& b = nodes.find(road.path[i])->second;
        len += distance(a, b);
    }
    return len;
}

map<string, double> get_road_lengths(
    const map<OSMID, Node>& nodes, const map<OSMID, Road>& roads) {
    map<string, double> road_lengths;
    for (const auto& osmid_road : roads) {
        const Road& road = osmid_road.second;
        road_lengths[road.name] += get_road_length(nodes, road);
    }
    return road_lengths;
}

struct Edge {
    pair<OSMID, OSMID> edge;
    double weight;

    Edge(OSMID first, OSMID second, pair<Node, Node> nodes) {
        edge = std::make_pair(first, second);
        weight = distance(nodes.first, nodes.second);
    }

    bool operator<(const Edge& rhs) const {
        return weight > rhs.weight;
    }
};

class MST {
private:
    priority_queue<Edge> q;
    map<OSMID, Node> nodes;
    map<OSMID, Road> roads;
    map<OSMID, OSMID> union_find;
    vector<Edge> mst;
public:
    MST() {
        nodes = load_nodes("nodes.txt");
        roads = load_roads("roads.txt");
        for (auto& node : nodes) {
            union_find[node.first] = node.first;
        }
    }
    
    void get_edges() {
        for (auto& road : roads) {
            for (size_t i = 0; i < road.second.path.size() - 1; i++) {
                if (road.second.path[i] != road.second.path[i + 1]) {
                    q.emplace(road.second.path[i], road.second.path[i + 1],
                        std::make_pair(nodes[road.second.path[i]], nodes[road.second.path[i + 1]]));
                }
            }
        }
    }

    void build_tree() {
        OSMID parent1, parent2, region1, region2;
        while (!q.empty()) {
			parent1 = q.top().edge.first;
			region1 = union_find[parent1];
			parent2 = q.top().edge.second;
			region2 = union_find[parent2];
            if (region1 != region2) {
                while (parent1 != region1) {
                    parent1 = region1;
                    region1 = union_find[parent1];
                }
                while (parent2 != region2) {
                    parent2 = region2;
                    region2 = union_find[parent2];
                }
                if (region1 != region2) {
                    mst.push_back(q.top());
                    if (region1 < region2) {
                        union_find[region2] = region1;
                    }
                    else { // region2 < region1
                        union_find[region1] = region2;
                    }
                }
            }
            q.pop();
        }
    }

    void output() {
        ofstream myfile;
        myfile.open("output.txt");
		double sum = 0;
		for (auto& edge : mst) {
			sum += edge.weight;
		}
		myfile << sum << endl;
        myfile << mst.size() << endl;
		for (auto& ele : mst) {
            myfile << ele.edge.first << " " << ele.edge.second << endl;
		}
        myfile.close();
    }
};

int main(int argc, char* argv[]) {
    MST mst;
    mst.get_edges();
    mst.build_tree();
    mst.output();

    return 0;
}