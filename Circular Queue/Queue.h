#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>


template <typename T>
class Queue {
public:
	std::vector<T> _data;
	size_t _head, _tail;
	static T _sentinel;
public:
	static const int MAX_DISP_ELEMS = 100;
	Queue(size_t size);
	static void set_sentinel(const T& elem) { _sentinel = elem; }
	static T get_sentinel() { return _sentinel; }
	bool is_empty() const;
	bool is_full() const;
	size_t size() const;
	void resize(size_t size);
	const T& peek() const;
	bool dequeue();
	bool enqueue(const T& elem);
	std::string to_string(size_t limit = MAX_DISP_ELEMS) const;
	template<typename T>
	friend std::ostream& operator<<(std::ostream& os, Queue<T>& q);
	//friend class Tests;
};
template <typename T> T Queue<T>::_sentinel = T();

template <typename T>
void popalot(Queue<T>& q);

template <typename T>
void popalot(Queue<T>& q) {
	while (q.dequeue()) {}
}

template <typename T>
std::ostream& operator<<(std::ostream& os, Queue<T>& q) {
	os << q.to_string();
	return os;
}

template <typename T>
size_t Queue<T>::size() const {
	size_t x = _tail - _head;
	if (x < 0) {
		return x *= -1;
	}
	else {
		return x;
	}
}


template <typename T>
Queue<T>::Queue(size_t size) {
	_data.resize(size + 1);
	_head = 0;
	_tail = 0;
}


template <typename T>
bool Queue<T>::enqueue(const T& elem) {
	if (is_full()) {
		return false;
	}
	_data[_tail] = elem;
	if (_tail == _data.size() - 1) {
		_tail = 0;
	}
	else {
		_tail++;
	}
	return true;
}


template <typename T>
bool Queue<T>::dequeue() {
	if (is_empty()){
		return false;
	}
	if (_head == _data.size() - 1) {
		_head = 0;
	}
	else {
		_head++;
	}
	return true;
}


template <typename T>
const T& Queue<T>::peek() const {
	if (is_empty()) {
		return _sentinel;
	}
	return _data[_head];
}


template <typename T>
bool Queue<T>::is_empty() const {
	return (_head == _tail);
}


template <typename T>
bool Queue<T>::is_full() const {
	return (_head == (_tail + 1) % _data.size());
}


template <typename T>
void Queue<T>::resize(size_t size) {
	Queue<T> newQ(size);

	while (!is_empty() && !newQ.is_full()) {
		newQ.enqueue(_data[_head]);
		dequeue();
	}

	_data = newQ._data;
	_head = newQ._head;
	_tail = newQ._tail;
}


template <typename T>
std::string Queue<T>::to_string(size_t lim) const {
	std::string stringification = "";
	std::stringstream ss;
	Queue<T> newQ(lim);
	size_t counter = 0;

	newQ._head = _head;
	newQ._tail = _tail;
	newQ._data = _data;

	stringification += "# Queue - size = " + std::to_string(size()) + "\n";
	stringification += "data : ";

	while (!newQ.is_empty() && counter < lim) {
		ss << newQ._data[newQ._head];
		stringification += ss.str() + " ";
		ss.str("");
		newQ.dequeue();
		counter++;
	}
	stringification += "\n";
	return stringification;
}