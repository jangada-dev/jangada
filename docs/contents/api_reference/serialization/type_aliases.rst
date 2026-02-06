**************
Type Aliases
**************

.. currentmodule:: jangada.serialization

SerializableProperty uses several type aliases to improve code readability and
provide clear type hints for callback functions.


Type Variable
=============

.. autodata:: T
   :annotation: = TypeVar('T')

   Type variable representing the type of the property value.


Callable Type Aliases
=====================

.. autodata:: DefaultCallable
   :no-value:

   Callable that produces a default value for a property.

   **Signature:** ``(instance: object) -> T``

   Parameters:
      - **instance** (*object*) -- The instance for which to generate the default value

   Returns:
      The default value of type T

   Example::

      def make_default_list(instance: object) -> list:
          return []

      class MyClass:
          items = SerializableProperty(default=make_default_list)


.. autodata:: Observer
   :no-value:

   Callable that observes property value changes.

   **Signature:** ``(instance: object, old_value: T, new_value: T) -> None``

   Parameters:
      - **instance** (*object*) -- The instance whose property changed
      - **old_value** (*T*) -- The previous value of the property
      - **new_value** (*T*) -- The new value of the property

   Returns:
      None

   Example::

      def log_changes(instance: object, old: Any, new: Any) -> None:
          print(f"Value changed from {old} to {new}")

      class MyClass:
          value = SerializableProperty(observers={log_changes})


.. autodata:: Parser
   :no-value:

   Callable that parses and validates property values.

   **Signature:** ``(instance: object, value: Any) -> T``

   Parameters:
      - **instance** (*object*) -- The instance for which to parse the value
      - **value** (*Any*) -- The raw value to parse

   Returns:
      The parsed and validated value of type T

   Raises:
      Any exception raised during parsing propagates to the caller

   Example::

      def parse_positive(instance: object, value: Any) -> float:
          result = float(value)
          if result <= 0:
              raise ValueError("Value must be positive")
          return result

      class MyClass:
          temperature = SerializableProperty(parser=parse_positive)


.. autodata:: Postinitializer
   :no-value:

   Callable that performs post-initialization setup.

   **Signature:** ``(instance: object) -> None``

   Parameters:
      - **instance** (*object*) -- The instance to initialize

   Returns:
      None

   Notes:
      The Postinitializer is called once after the property is first set or accessed.
      It runs after the value has been stored and observers have been called.

   Example::

      def setup_data(instance: object) -> None:
          print("Loading expensive data...")
          if instance.data is None:
              instance.data = load_from_disk()

      class MyClass:
          data = SerializableProperty(postPostinitializer=setup_data)