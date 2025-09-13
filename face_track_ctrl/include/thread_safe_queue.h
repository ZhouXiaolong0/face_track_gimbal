#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_THREAD_SAFE_QUEUE_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_THREAD_SAFE_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

/**
 * @brief 线程安全的队列模板类。
 *
 * 该类使用互斥锁和条件变量实现线程安全的队列操作，
 * 支持阻塞 pop、非阻塞 tryPop，以及队列大小限制。
 *
 * @tparam T 队列中存储的数据类型。
 */
template <typename T> class ThreadSafeQueue {
private:
  /// 内部队列
  std::queue<T> queue_;

  /// 互斥锁
  std::mutex mtx_;

  /// 条件变量
  std::condition_variable cv_;

  /// 队列最大容量
  size_t max_size_;

public:
  /**
   * @brief 构造函数
   * @param max_size 队列最大容量，默认值为 10
   */
  explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

  /**
   * @brief 向队列中添加一个元素。
   *
   * 如果队列已满，会先弹出队头元素，然后插入新元素。
   * 插入元素后会通知可能正在等待的线程。
   *
   * @param item 要插入的元素
   */
  void push(const T &item) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (queue_.size() >= max_size_)
      queue_.pop();
    queue_.push(item);
    cv_.notify_one();
  }

  /**
   * @brief 从队列中弹出一个元素（阻塞）。
   *
   * 如果队列为空，该函数会阻塞直到有元素可弹出。
   *
   * @return 队头元素
   */
  T pop() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    T item = queue_.front();
    queue_.pop();
    return item;
  }

  /**
   * @brief 尝试从队列中弹出一个元素（非阻塞）。
   *
   * 如果队列为空，返回 false；否则返回 true 并将弹出的元素赋值给 item。
   *
   * @param item 弹出的元素的引用
   * @return 成功弹出返回 true，否则返回 false
   */
  bool tryPop(T &item) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty())
      return false;
    item = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   * @brief 获取队列当前元素数量
   *
   * @return 队列大小
   */
  size_t size() {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
  }
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_THREAD_SAFE_QUEUE_H_