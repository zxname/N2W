#pragma once
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>

template <typename T>
class FixedSizeQueue {
public:
	FixedSizeQueue();
	FixedSizeQueue(size_t queue_max_size);
	~FixedSizeQueue();
	void Set_Max_Size(size_t queue_max_size);
	void Push_Back(const T& item);
	T Pop_Front();
	T Pop_Back();
	T Get_Front();
	T Get_Back();
	bool Is_Empty();
	size_t Size();
	std::deque<T> queue;

private:
	mutable std::mutex mtx;
	size_t queue_max_size;
};

template <typename T>
FixedSizeQueue<T>::FixedSizeQueue()
{
	;
}

template <typename T>
FixedSizeQueue<T>::FixedSizeQueue(size_t queue_max_size) {
	this->queue_max_size = queue_max_size;
}
template <typename T>
FixedSizeQueue<T>::~FixedSizeQueue() {
	std::lock_guard<std::mutex> lock(this->mtx);
	this->queue.clear();
}

template <typename T>
void FixedSizeQueue<T>::Set_Max_Size(size_t queue_max_size) {
	std::lock_guard<std::mutex> lock(this->mtx);
	this->queue_max_size = queue_max_size;
}

template <typename T>
void FixedSizeQueue<T>::Push_Back(const T& item) {
	std::lock_guard<std::mutex> lock(this->mtx);
	if (this->queue.size() == this->queue_max_size) {
		this->queue.pop_front();  // If the queue is full, remove the oldest frame
	}
	this->queue.push_back(item);
}

template <typename T>
bool FixedSizeQueue<T>::Is_Empty() {
	std::lock_guard<std::mutex> lock(this->mtx);
	return this->queue.empty();
}

template <typename T>
size_t FixedSizeQueue<T>::Size() {
	std::lock_guard<std::mutex> lock(this->mtx);
	return this->queue.size();
}

template <typename T>
T FixedSizeQueue<T>::Pop_Front() {
	std::lock_guard<std::mutex> lock(this->mtx);
	if (this->queue.empty()) {
		return T();
	}
	else {
		T tmp = this->queue.front();
		this->queue.pop_front();
		return tmp;
	}
}

template <typename T>
T FixedSizeQueue<T>::Pop_Back() {
	std::lock_guard<std::mutex> lock(this->mtx);
	if (this->queue.empty()) {
		return T();
	}
	else {
		T tmp = this->queue.back();
		this->queue.pop_back();
		return tmp;
	}
}

template <typename T>
T FixedSizeQueue<T>::Get_Front() {
	std::lock_guard<std::mutex> lock(this->mtx);
	if (this->queue.empty()) {
		return T();
	}
	else {
		T tmp = this->queue.front();
		return tmp;
	}
}

template <typename T>
T FixedSizeQueue<T>::Get_Back() {
	std::lock_guard<std::mutex> lock(this->mtx);
	if (this->queue.empty()) {
		return T();
	}
	else {
		T tmp = this->queue.back();
		return tmp;
	}
}