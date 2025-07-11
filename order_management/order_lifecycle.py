"""
Order lifecycle management and state machine.
"""

from typing import Dict, List, Optional, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass

from .order_types import Order, OrderStatus
from utils.logging import get_logger

logger = get_logger(__name__)


class OrderState(Enum):
    """Order states in the lifecycle."""
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"


class OrderTransition(Enum):
    """Order state transitions."""
    VALIDATE = "validate"
    SUBMIT = "submit"
    ACKNOWLEDGE = "acknowledge"
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    CANCEL = "cancel"
    REJECT = "reject"
    EXPIRE = "expire"
    REPLACE = "replace"


@dataclass
class StateTransition:
    """State transition definition."""
    from_state: OrderState
    to_state: OrderState
    transition: OrderTransition
    condition: Optional[Callable[[Order], bool]] = None
    action: Optional[Callable[[Order], None]] = None


class OrderStateMachine:
    """State machine for order lifecycle management."""
    
    def __init__(self):
        self.transitions: Dict[tuple, StateTransition] = {}
        self.state_handlers: Dict[OrderState, List[Callable]] = {}
        self._setup_transitions()
    
    def _setup_transitions(self) -> None:
        """Setup valid state transitions."""
        transitions = [
            # From CREATED
            StateTransition(OrderState.CREATED, OrderState.VALIDATED, OrderTransition.VALIDATE),
            StateTransition(OrderState.CREATED, OrderState.REJECTED, OrderTransition.REJECT),
            StateTransition(OrderState.CREATED, OrderState.CANCELED, OrderTransition.CANCEL),
            
            # From VALIDATED
            StateTransition(OrderState.VALIDATED, OrderState.SUBMITTED, OrderTransition.SUBMIT),
            StateTransition(OrderState.VALIDATED, OrderState.REJECTED, OrderTransition.REJECT),
            StateTransition(OrderState.VALIDATED, OrderState.CANCELED, OrderTransition.CANCEL),
            
            # From SUBMITTED
            StateTransition(OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderTransition.ACKNOWLEDGE),
            StateTransition(OrderState.SUBMITTED, OrderState.REJECTED, OrderTransition.REJECT),
            StateTransition(OrderState.SUBMITTED, OrderState.CANCELED, OrderTransition.CANCEL),
            
            # From ACKNOWLEDGED
            StateTransition(OrderState.ACKNOWLEDGED, OrderState.WORKING, OrderTransition.ACKNOWLEDGE),
            StateTransition(OrderState.ACKNOWLEDGED, OrderState.FILLED, OrderTransition.COMPLETE_FILL),
            StateTransition(OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderTransition.PARTIAL_FILL),
            StateTransition(OrderState.ACKNOWLEDGED, OrderState.CANCELED, OrderTransition.CANCEL),
            StateTransition(OrderState.ACKNOWLEDGED, OrderState.EXPIRED, OrderTransition.EXPIRE),
            
            # From WORKING
            StateTransition(OrderState.WORKING, OrderState.FILLED, OrderTransition.COMPLETE_FILL),
            StateTransition(OrderState.WORKING, OrderState.PARTIALLY_FILLED, OrderTransition.PARTIAL_FILL),
            StateTransition(OrderState.WORKING, OrderState.CANCELED, OrderTransition.CANCEL),
            StateTransition(OrderState.WORKING, OrderState.EXPIRED, OrderTransition.EXPIRE),
            StateTransition(OrderState.WORKING, OrderState.REPLACED, OrderTransition.REPLACE),
            
            # From PARTIALLY_FILLED
            StateTransition(OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderTransition.COMPLETE_FILL),
            StateTransition(OrderState.PARTIALLY_FILLED, OrderState.CANCELED, OrderTransition.CANCEL),
            StateTransition(OrderState.PARTIALLY_FILLED, OrderState.EXPIRED, OrderTransition.EXPIRE),
            StateTransition(OrderState.PARTIALLY_FILLED, OrderState.REPLACED, OrderTransition.REPLACE),
        ]
        
        for transition in transitions:
            key = (transition.from_state, transition.transition)
            self.transitions[key] = transition
    
    def can_transition(self, current_state: OrderState, transition: OrderTransition) -> bool:
        """Check if transition is valid from current state."""
        return (current_state, transition) in self.transitions
    
    def transition_order(
        self,
        order: Order,
        transition: OrderTransition,
        current_state: OrderState
    ) -> OrderState:
        """Transition order to new state."""
        if not self.can_transition(current_state, transition):
            raise ValueError(
                f"Invalid transition {transition.value} from state {current_state.value}"
            )
        
        state_transition = self.transitions[(current_state, transition)]
        new_state = state_transition.to_state
        
        # Check condition if present
        if state_transition.condition and not state_transition.condition(order):
            raise ValueError(
                f"Transition condition not met for {transition.value} from {current_state.value}"
            )
        
        # Execute action if present
        if state_transition.action:
            state_transition.action(order)
        
        # Execute state handlers
        self._execute_state_handlers(new_state, order)
        
        logger.info(
            f"Order {order.order_id} transitioned: {current_state.value} -> {new_state.value} "
            f"via {transition.value}"
        )
        
        return new_state
    
    def add_state_handler(self, state: OrderState, handler: Callable[[Order], None]) -> None:
        """Add handler for when order enters a specific state."""
        if state not in self.state_handlers:
            self.state_handlers[state] = []
        self.state_handlers[state].append(handler)
    
    def _execute_state_handlers(self, state: OrderState, order: Order) -> None:
        """Execute handlers for a state."""
        if state in self.state_handlers:
            for handler in self.state_handlers[state]:
                try:
                    handler(order)
                except Exception as e:
                    logger.error(f"Error in state handler for {state.value}: {e}")
    
    def get_valid_transitions(self, current_state: OrderState) -> List[OrderTransition]:
        """Get valid transitions from current state."""
        valid_transitions = []
        for (state, transition), _ in self.transitions.items():
            if state == current_state:
                valid_transitions.append(transition)
        return valid_transitions
    
    def get_terminal_states(self) -> Set[OrderState]:
        """Get terminal states (no outgoing transitions)."""
        all_from_states = {state for state, _ in self.transitions.keys()}
        all_to_states = {trans.to_state for trans in self.transitions.values()}
        
        # States that appear in to_states but not in from_states are terminal
        terminal_states = all_to_states - all_from_states
        
        # Add known terminal states
        terminal_states.update({
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED
        })
        
        return terminal_states


class OrderLifecycleManager:
    """Manager for order lifecycle operations."""
    
    def __init__(self):
        self.state_machine = OrderStateMachine()
        self.order_states: Dict[str, OrderState] = {}
        self.lifecycle_events: Dict[str, List[Dict]] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default state handlers."""
        self.state_machine.add_state_handler(
            OrderState.SUBMITTED,
            self._on_order_submitted
        )
        self.state_machine.add_state_handler(
            OrderState.FILLED,
            self._on_order_filled
        )
        self.state_machine.add_state_handler(
            OrderState.CANCELED,
            self._on_order_canceled
        )
        self.state_machine.add_state_handler(
            OrderState.REJECTED,
            self._on_order_rejected
        )
    
    def register_order(self, order: Order) -> None:
        """Register order for lifecycle management."""
        self.order_states[order.order_id] = OrderState.CREATED
        self.lifecycle_events[order.order_id] = []
        
        self._record_event(order.order_id, OrderState.CREATED, "Order created")
        
        logger.info(f"Order {order.order_id} registered for lifecycle management")
    
    def transition_order(
        self,
        order: Order,
        transition: OrderTransition,
        notes: Optional[str] = None
    ) -> bool:
        """Transition order to new state."""
        if order.order_id not in self.order_states:
            self.register_order(order)
        
        current_state = self.order_states[order.order_id]
        
        try:
            new_state = self.state_machine.transition_order(order, transition, current_state)
            self.order_states[order.order_id] = new_state
            
            # Update order status to match state
            self._update_order_status(order, new_state)
            
            # Record event
            self._record_event(
                order.order_id,
                new_state,
                notes or f"Transitioned via {transition.value}"
            )
            
            return True
            
        except ValueError as e:
            logger.error(f"Failed to transition order {order.order_id}: {e}")
            return False
    
    def _update_order_status(self, order: Order, state: OrderState) -> None:
        """Update order status based on state."""
        state_to_status = {
            OrderState.CREATED: OrderStatus.PENDING,
            OrderState.VALIDATED: OrderStatus.PENDING,
            OrderState.SUBMITTED: OrderStatus.SUBMITTED,
            OrderState.ACKNOWLEDGED: OrderStatus.ACKNOWLEDGED,
            OrderState.WORKING: OrderStatus.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            OrderState.FILLED: OrderStatus.FILLED,
            OrderState.CANCELED: OrderStatus.CANCELED,
            OrderState.REJECTED: OrderStatus.REJECTED,
            OrderState.EXPIRED: OrderStatus.EXPIRED,
            OrderState.REPLACED: OrderStatus.REPLACED
        }
        
        if state in state_to_status:
            order.status = state_to_status[state]
    
    def _record_event(self, order_id: str, state: OrderState, notes: str) -> None:
        """Record lifecycle event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'state': state.value,
            'notes': notes
        }
        
        if order_id not in self.lifecycle_events:
            self.lifecycle_events[order_id] = []
        
        self.lifecycle_events[order_id].append(event)
    
    def get_order_state(self, order_id: str) -> Optional[OrderState]:
        """Get current state of order."""
        return self.order_states.get(order_id)
    
    def get_order_history(self, order_id: str) -> List[Dict]:
        """Get lifecycle history for order."""
        return self.lifecycle_events.get(order_id, [])
    
    def get_valid_transitions(self, order_id: str) -> List[OrderTransition]:
        """Get valid transitions for order."""
        current_state = self.order_states.get(order_id)
        if current_state:
            return self.state_machine.get_valid_transitions(current_state)
        return []
    
    def is_terminal_state(self, order_id: str) -> bool:
        """Check if order is in terminal state."""
        current_state = self.order_states.get(order_id)
        if current_state:
            return current_state in self.state_machine.get_terminal_states()
        return False
    
    def get_orders_by_state(self, state: OrderState) -> List[str]:
        """Get order IDs in specific state."""
        return [
            order_id for order_id, order_state in self.order_states.items()
            if order_state == state
        ]
    
    def cleanup_terminal_orders(self, retention_hours: int = 24) -> int:
        """Clean up old terminal orders."""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        cleaned_count = 0
        
        terminal_states = self.state_machine.get_terminal_states()
        orders_to_remove = []
        
        for order_id, state in self.order_states.items():
            if state in terminal_states:
                events = self.lifecycle_events.get(order_id, [])
                if events:
                    last_event_time = datetime.fromisoformat(events[-1]['timestamp'])
                    if last_event_time < cutoff_time:
                        orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del self.order_states[order_id]
            del self.lifecycle_events[order_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} terminal orders older than {retention_hours} hours")
        
        return cleaned_count
    
    def get_lifecycle_statistics(self) -> Dict[str, any]:
        """Get lifecycle statistics."""
        state_counts = {}
        for state in OrderState:
            state_counts[state.value] = len(self.get_orders_by_state(state))
        
        total_orders = len(self.order_states)
        terminal_orders = sum(
            len(self.get_orders_by_state(state))
            for state in self.state_machine.get_terminal_states()
        )
        
        return {
            'total_orders': total_orders,
            'active_orders': total_orders - terminal_orders,
            'terminal_orders': terminal_orders,
            'state_distribution': state_counts,
            'terminal_states': [state.value for state in self.state_machine.get_terminal_states()]
        }
    
    # Default event handlers
    def _on_order_submitted(self, order: Order) -> None:
        """Handler for when order is submitted."""
        order.submitted_at = datetime.now()
        logger.info(f"Order {order.order_id} submitted to broker/exchange")
    
    def _on_order_filled(self, order: Order) -> None:
        """Handler for when order is filled."""
        order.filled_at = datetime.now()
        logger.info(f"Order {order.order_id} completely filled")
    
    def _on_order_canceled(self, order: Order) -> None:
        """Handler for when order is canceled."""
        order.canceled_at = datetime.now()
        logger.info(f"Order {order.order_id} canceled")
    
    def _on_order_rejected(self, order: Order) -> None:
        """Handler for when order is rejected."""
        logger.warning(f"Order {order.order_id} rejected: {order.rejection_reason}")


# Global lifecycle manager instance
_lifecycle_manager = None


def get_lifecycle_manager() -> OrderLifecycleManager:
    """Get global order lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = OrderLifecycleManager()
    return _lifecycle_manager