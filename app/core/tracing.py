import asyncio
import logging
from typing import Dict, Any, Optional, List
import structlog
from datetime import datetime
import uuid
import json

logger = structlog.get_logger()

class TraceSpan:
    def __init__(self, trace_id: str, span_id: str, operation_name: str, 
                 parent_id: Optional[str] = None, tags: Dict[str, Any] = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.duration_ms = None
        self.logs = []
        self.status = "started"
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def finish(self, status: str = "success", error: Optional[str] = None):
        """Finish the span"""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        
        if error:
            self.add_log(f"Error: {error}", level="error")
            self.tags["error"] = True
            self.tags["error.message"] = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation_name": self.operation_name,
            "tags": self.tags,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "logs": self.logs
        }

class TraceContext:
    def __init__(self, trace_id: str = None, parent_span_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.spans = []
        self.current_span = None
    
    def create_span(self, operation_name: str, tags: Dict[str, Any] = None) -> TraceSpan:
        """Create a new span"""
        span_id = str(uuid.uuid4())
        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=span_id,
            operation_name=operation_name,
            parent_id=self.parent_span_id,
            tags=tags or {}
        )
        self.spans.append(span)
        self.current_span = span
        return span
    
    def get_trace_id(self) -> str:
        """Get the trace ID"""
        return self.trace_id
    
    def get_current_span(self) -> Optional[TraceSpan]:
        """Get the current active span"""
        return self.current_span
    
    def finish_current_span(self, status: str = "success", error: Optional[str] = None):
        """Finish the current span"""
        if self.current_span:
            self.current_span.finish(status, error)
            self.current_span = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary"""
        return {
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "spans": [span.to_dict() for span in self.spans],
            "span_count": len(self.spans)
        }

class TracingService:
    def __init__(self):
        self.active_traces = {}
        self.trace_storage = []
        self.max_traces = 1000
    
    def start_trace(self, operation_name: str, trace_id: str = None, 
                   parent_span_id: str = None, tags: Dict[str, Any] = None) -> TraceContext:
        """Start a new trace"""
        context = TraceContext(trace_id, parent_span_id)
        context.create_span(operation_name, tags)
        
        # Store active trace
        self.active_traces[context.trace_id] = context
        
        logger.info(f"Started trace: {context.trace_id}", operation=operation_name)
        return context
    
    def finish_trace(self, trace_id: str, status: str = "success", error: Optional[str] = None):
        """Finish a trace"""
        if trace_id in self.active_traces:
            context = self.active_traces[trace_id]
            context.finish_current_span(status, error)
            
            # Move to storage
            self.trace_storage.append(context.to_dict())
            
            # Cleanup old traces
            if len(self.trace_storage) > self.max_traces:
                self.trace_storage = self.trace_storage[-self.max_traces:]
            
            # Remove from active
            del self.active_traces[trace_id]
            
            logger.info(f"Finished trace: {trace_id}", status=status)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID"""
        # Check active traces first
        if trace_id in self.active_traces:
            return self.active_traces[trace_id].to_dict()
        
        # Check stored traces
        for trace in self.trace_storage:
            if trace["trace_id"] == trace_id:
                return trace
        
        return None
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces"""
        return self.trace_storage[-limit:]
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get trace statistics"""
        total_traces = len(self.trace_storage)
        active_traces = len(self.active_traces)
        
        if total_traces == 0:
            return {
                "total_traces": 0,
                "active_traces": active_traces,
                "average_duration_ms": 0,
                "success_rate": 0
            }
        
        # Calculate statistics
        durations = []
        success_count = 0
        
        for trace in self.trace_storage:
            for span in trace["spans"]:
                if span["duration_ms"]:
                    durations.append(span["duration_ms"])
                if span["status"] == "success":
                    success_count += 1
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        success_rate = success_count / total_traces if total_traces > 0 else 0
        
        return {
            "total_traces": total_traces,
            "active_traces": active_traces,
            "average_duration_ms": avg_duration,
            "success_rate": success_rate,
            "total_spans": sum(len(trace["spans"]) for trace in self.trace_storage)
        }

# Global tracing service
tracing_service = TracingService()

def trace_operation(operation_name: str, tags: Dict[str, Any] = None):
    """Decorator for tracing operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Start trace
            context = tracing_service.start_trace(operation_name, tags=tags)
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Finish trace
                tracing_service.finish_trace(context.get_trace_id(), "success")
                
                return result
                
            except Exception as e:
                # Finish trace with error
                tracing_service.finish_trace(context.get_trace_id(), "error", str(e))
                raise
        
        def sync_wrapper(*args, **kwargs):
            # Start trace
            context = tracing_service.start_trace(operation_name, tags=tags)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Finish trace
                tracing_service.finish_trace(context.get_trace_id(), "success")
                
                return result
                
            except Exception as e:
                # Finish trace with error
                tracing_service.finish_trace(context.get_trace_id(), "error", str(e))
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class QueryTracer:
    def __init__(self, tracing_service: TracingService):
        self.tracing_service = tracing_service
    
    def trace_query(self, question: str, collection: str, search_type: str = "hybrid"):
        """Create a trace for a query operation"""
        tags = {
            "operation": "query",
            "question_length": len(question),
            "collection": collection,
            "search_type": search_type
        }
        
        return self.tracing_service.start_trace("query_operation", tags=tags)
    
    def trace_retrieval(self, trace_id: str, method: str, result_count: int):
        """Add retrieval span to trace"""
        if trace_id in self.tracing_service.active_traces:
            context = self.tracing_service.active_traces[trace_id]
            span = context.create_span("retrieval", {
                "method": method,
                "result_count": result_count
            })
            return span
        return None
    
    def trace_reranking(self, trace_id: str, input_count: int, output_count: int):
        """Add reranking span to trace"""
        if trace_id in self.tracing_service.active_traces:
            context = self.tracing_service.active_traces[trace_id]
            span = context.create_span("reranking", {
                "input_count": input_count,
                "output_count": output_count
            })
            return span
        return None
    
    def trace_generation(self, trace_id: str, model: str, token_count: int):
        """Add generation span to trace"""
        if trace_id in self.tracing_service.active_traces:
            context = self.tracing_service.active_traces[trace_id]
            span = context.create_span("generation", {
                "model": model,
                "token_count": token_count
            })
            return span
        return None

# Global query tracer
query_tracer = QueryTracer(tracing_service)

def get_trace_context() -> Optional[TraceContext]:
    """Get current trace context (simplified)"""
    # In a real implementation, this would get the current context from thread local storage
    return None

def inject_trace_headers(trace_id: str) -> Dict[str, str]:
    """Inject trace headers for distributed tracing"""
    return {
        "X-Trace-ID": trace_id,
        "X-Trace-Span": str(uuid.uuid4())
    }

def extract_trace_headers(headers: Dict[str, str]) -> Optional[TraceContext]:
    """Extract trace context from headers"""
    trace_id = headers.get("X-Trace-ID")
    parent_span_id = headers.get("X-Trace-Span")
    
    if trace_id:
        return TraceContext(trace_id, parent_span_id)
    
    return None
