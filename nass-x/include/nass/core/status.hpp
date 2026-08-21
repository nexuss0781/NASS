#ifndef NASS_CORE_STATUS_HPP
#define NASS_CORE_STATUS_HPP

#include <string>
#include <stdexcept>
#include <variant>
#include "nass/core/types.hpp"

namespace nass_x::tensor {

// Result type for operations that can fail
template<typename T>
class Result {
public:
    // Success constructor
    explicit Result(T value) 
        : data_(std::move(value))
        , status_(Status::OK) {}
    
    // Error constructor
    explicit Result(Status status, const std::string& error_msg = "")
        : data_(T{})
        , status_(status)
        , error_msg_(error_msg) {}
    
    // Check if result is OK
    bool ok() const { return status_ == Status::OK; }
    
    // Get value (undefined behavior if not OK)
    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    
    // Get status
    Status status() const { return status_; }
    
    // Get error message
    const std::string& error_message() const { return error_msg_; }
    
    // Access operators
    T* operator->() { return &std::get<T>(data_); }
    const T* operator->() const { return &std::get<T>(data_); }
    T& operator*() { return std::get<T>(data_); }
    const T& operator*() const { return std::get<T>(data_); }
    
private:
    std::variant<T> data_;
    Status status_;
    std::string error_msg_;
};

// Specialization for void results
template<>
class Result<void> {
public:
    explicit Result(Status status = Status::OK, const std::string& error_msg = "")
        : status_(status)
        , error_msg_(error_msg) {}
    
    bool ok() const { return status_ == Status::OK; }
    Status status() const { return status_; }
    const std::string& error_message() const { return error_msg_; }
    
private:
    Status status_;
    std::string error_msg_;
};

// Exception class for NASS errors
class NASSException : public std::runtime_error {
public:
    explicit NASSException(const std::string& msg) 
        : std::runtime_error(msg) {}
    
    NASSException(Status status, const std::string& msg)
        : std::runtime_error(format_message(status, msg))
        , status_(status) {}
    
    Status status() const { return status_; }
    
private:
    Status status_ = Status::ERROR_INVALID_PARAM;
    
    static std::string format_message(Status status, const std::string& msg) {
        std::string prefix;
        switch (status) {
            case Status::OK: return msg;
            case Status::ERROR_INVALID_PARAM: prefix = "Invalid parameter"; break;
            case Status::ERROR_OUT_OF_MEMORY: prefix = "Out of memory"; break;
            case Status::ERROR_FILE_NOT_FOUND: prefix = "File not found"; break;
            case Status::ERROR_UNSUPPORTED_FORMAT: prefix = "Unsupported format"; break;
            case Status::ERROR_FFT_FAILED: prefix = "FFT computation failed"; break;
            case Status::ERROR_THREAD_FAILED: prefix = "Thread operation failed"; break;
            default: prefix = "Unknown error"; break;
        }
        return prefix + (msg.empty() ? "" : ": " + msg);
    }
};

// Helper macros for error handling
#define NASS_THROW(status, msg) throw NASSException(status, msg)
#define NASS_CHECK(condition, status, msg) \
    do { if (!(condition)) NASS_THROW(status, msg); } while(0)
#define NASS_REQUIRE(condition, msg) NASS_CHECK(condition, Status::ERROR_INVALID_PARAM, msg)

// Status code to string conversion
inline std::string status_to_string(Status status) {
    switch (status) {
        case Status::OK: return "OK";
        case Status::ERROR_INVALID_PARAM: return "Invalid Parameter";
        case Status::ERROR_OUT_OF_MEMORY: return "Out of Memory";
        case Status::ERROR_FILE_NOT_FOUND: return "File Not Found";
        case Status::ERROR_UNSUPPORTED_FORMAT: return "Unsupported Format";
        case Status::ERROR_FFT_FAILED: return "FFT Failed";
        case Status::ERROR_THREAD_FAILED: return "Thread Failed";
        case Status::WARNING_PRECISION_LOSS: return "Warning: Precision Loss";
        default: return "Unknown Status";
    }
}

} // namespace nass_x::tensor

#endif // NASS_CORE_STATUS_HPP
