#include "gotorch/kernels.hpp"

template <typename T>
ndarray<T> ndarray<T>::add(const ndarray<T>& addend) const {
    return gotorch::kernels::add(*this, addend);
}

template <typename T>
ndarray<T> ndarray<T>::sub(const ndarray<T>& subtrahend) const {
    return gotorch::kernels::sub(*this, subtrahend);
}

template <typename T>
ndarray<T> ndarray<T>::matmul(const ndarray<T>& other) const {
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::relu() const {
    return gotorch::kernels::relu(*this);
}

template <typename T>
ndarray<T> ndarray<T>::relu_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::relu_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::tanh() const {
    return gotorch::kernels::tanh(*this);
}

template <typename T>
ndarray<T> ndarray<T>::tanh_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::tanh_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::sigmoid() const {
    return gotorch::kernels::sigmoid(*this);
}

template <typename T>
ndarray<T> ndarray<T>::sigmoid_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::sigmoid_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::leaky_relu(T alpha) const {
    return gotorch::kernels::leaky_relu(*this, alpha);
}

template <typename T>
ndarray<T> ndarray<T>::leaky_relu_backward(const ndarray<T>& grad_output, T alpha) const {
    return gotorch::kernels::leaky_relu_backward(*this, grad_output, alpha);
}

template <typename T>
ndarray<T> ndarray<T>::sum(int dim, bool keepdim) const {
    return gotorch::kernels::sum(*this, dim, keepdim);
}

template <typename T>
ndarray<T> ndarray<T>::sum() const {
    return gotorch::kernels::sum(*this);
}

// Explicit template instantiations
template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int>;
