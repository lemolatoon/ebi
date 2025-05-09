/// A trait for writing individual bits, bytes, or a sequence of bits to a buffer.
///
/// This trait provides methods to write bits to a buffer and query the state of the buffer.
/// It is useful for applications that require precise control over binary data, such as
/// in data compression, serialization, or communication protocols.
pub trait BitWrite {
    /// Writes a single bit to the buffer.
    ///
    /// # Parameters
    ///
    /// - `bit`: The bit to be written. `true` represents a 1 bit, and `false` represents a 0 bit.
    fn write_bit(&mut self, bit: bool);

    /// Writes a single byte (8 bits) to the buffer.
    ///
    /// # Parameters
    ///
    /// - `byte`: The byte to be written.
    ///
    /// # Behavior
    ///
    /// This method writes the byte in its entirety, regardless of the current bit position.
    /// The internal cursor is advanced by 8 bits.
    fn write_byte(&mut self, byte: u8);

    /// Writes the lower `num` bits of the `bits` value to the buffer.
    ///
    /// # Parameters
    ///
    /// - `bits`: The value from which the bits will be written.
    /// - `num`: The number of bits to write. This value must be less than or equal to 64.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bits(0b1101, 4);
    /// let bytes = writer.close();
    ///
    /// assert_eq!(bytes.as_ref(), &[0b11010000]);
    /// ```
    #[inline]
    fn write_bits(&mut self, bits: u64, num: u32) {
        let mut remaining_bits = num;

        // Write full bytes if possible
        while remaining_bits >= 8 {
            let byte = (bits >> (remaining_bits - 8)) as u8;
            self.write_byte(byte);
            remaining_bits -= 8;
        }

        // Write any remaining bits
        if remaining_bits > 0 {
            let remaining_bits_value = bits & ((1 << remaining_bits) - 1);
            for i in (0..remaining_bits).rev() {
                let bit = (remaining_bits_value >> i) & 1 != 0;
                self.write_bit(bit);
            }
        }
    }

    /// Sets the cursor to a specific bit position within the buffer.
    ///
    /// # Parameters
    ///
    /// - `bit_cursor`: The bit position where the cursor should be set.
    ///
    /// # Behavior
    ///
    /// The cursor is moved to the specified position, allowing subsequent writes to
    /// overwrite bits at that position. This method does not expand the buffer.
    fn set_cursor_at_in_bit(&mut self, bit_cursor: usize);

    /// Returns the number of bits written to the buffer.
    ///
    /// # Returns
    ///
    /// - The total number of bits that have been written to the buffer.
    fn number_of_bits_written(&self) -> usize;

    /// Returns a slice of the underlying byte buffer.
    ///
    /// # Returns
    ///
    /// - A slice of the byte buffer containing the written bits.
    ///
    /// # Behavior
    ///
    /// The slice may include partially written bytes. For example, if only 6 bits
    /// have been written, the last byte in the slice will contain 2 unused bits.
    fn as_slice(&self) -> &[u8];

    /// Resets the writer, clearing the buffer and setting the cursor to the beginning.
    ///
    /// # Behavior
    ///
    /// This method allows the writer to be reused without creating a new instance.
    /// All previously written data is discarded.
    fn reset(&mut self);
}

/// A buffered bit writer that implements the `BitWrite` trait.
///
/// `BufferedBitWriter` allows writing individual bits, bytes, or sequences of bits
/// into an internal buffer with fine-grained control over the bit-level operations.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BufferedBitWriter {
    /// The internal buffer where bits and bytes are stored.
    ///
    /// This buffer is dynamically resized as bits are written to it, storing the written
    /// data in a byte-aligned manner.
    buffer: Vec<u8>,
    /// The current position within the byte (0 to 7) where the next bit will be written.
    ///
    /// If `bit_pos` is 8, it indicates that a new byte is needed for writing. The position
    /// is reset to 0 after each byte is filled.
    bit_pos: usize,
}

impl BufferedBitWriter {
    /// Creates a new `BufferedBitWriter`.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_pos: 8,
        }
    }

    /// Creates a new `BufferedBitWriter` with a specified capacity.
    ///
    /// This method allows you to pre-allocate space in the internal buffer, which can improve
    /// performance by reducing the need for reallocations as bits are written.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            bit_pos: 8,
        }
    }

    /// Creates a new `BufferedBitWriter` from an existing vector.
    ///
    /// This method takes ownership of an existing vector, clears its contents, and
    /// uses it as the internal buffer for the `BufferedBitWriter`. This can be useful
    /// if you have a pre-allocated vector that you want to reuse.
    pub fn from_vec(mut v: Vec<u8>) -> Self {
        v.clear();
        Self {
            buffer: v,
            bit_pos: 8,
        }
    }

    /// Finalizes the writing process and returns the buffer as a boxed slice.
    ///
    /// # Returns
    ///
    /// - A boxed slice containing the written bits.
    ///
    /// # Behavior
    ///
    /// After calling this method, the writer is closed, and no further writes are allowed.
    /// The returned buffer may include partially filled bytes with unused bits set to 0.
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bits(0b1011, 4);
    /// let buffer = writer.close();
    ///
    /// assert_eq!(buffer.as_ref(), &[0b10110000]);
    /// ```
    pub fn close(self) -> Box<[u8]> {
        self.buffer.into_boxed_slice()
    }
}

impl Default for BufferedBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: BitWrite> BitWrite for &mut W {
    fn write_bit(&mut self, bit: bool) {
        <W as BitWrite>::write_bit(self, bit);
    }

    fn write_byte(&mut self, byte: u8) {
        <W as BitWrite>::write_byte(self, byte);
    }

    fn write_bits(&mut self, bits: u64, num: u32) {
        <W as BitWrite>::write_bits(self, bits, num);
    }

    fn set_cursor_at_in_bit(&mut self, bit_cursor: usize) {
        <W as BitWrite>::set_cursor_at_in_bit(self, bit_cursor);
    }

    fn number_of_bits_written(&self) -> usize {
        <W as BitWrite>::number_of_bits_written(self)
    }

    fn as_slice(&self) -> &[u8] {
        <W as BitWrite>::as_slice(self)
    }

    fn reset(&mut self) {
        <W as BitWrite>::reset(self)
    }
}

impl BitWrite for BufferedBitWriter {
    /// Writes a single bit to the buffer.
    ///
    /// # Parameters
    ///
    /// - `bit`: The bit to be written. `true` represents a 1 bit, and `false` represents a 0 bit.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bit(true);
    /// let bytes = writer.close();
    ///
    /// assert_eq!(bytes.as_ref(), &[0b10000000]);
    /// ```
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        if self.bit_pos == 8 {
            self.buffer.push(0);
            self.bit_pos = 0;
        }

        if bit {
            let byte_pos = self.buffer.len() - 1;
            self.buffer[byte_pos] |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
    }

    /// Writes a single byte (8 bits) to the buffer.
    ///
    /// # Parameters
    ///
    /// - `byte`: The byte to be written.
    ///
    /// # Behavior
    ///
    /// This method writes the byte in its entirety, regardless of the current bit position.
    /// The internal cursor is advanced by 8 bits.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_byte(0xAA);
    /// let bytes = writer.close();
    ///
    /// assert_eq!(bytes.as_ref(), &[0xAA]);
    /// ```
    #[inline]
    fn write_byte(&mut self, byte: u8) {
        if self.bit_pos == 8 {
            self.buffer.push(byte);
        } else {
            let byte_pos = self.buffer.len() - 1;
            self.buffer[byte_pos] |= byte >> self.bit_pos;
            self.buffer.push(byte << (8 - self.bit_pos));
        }
    }

    /// Sets the cursor to a specific bit position within the buffer.
    ///
    /// # Parameters
    ///
    /// - `bit_cursor`: The bit position where the cursor should be set.
    ///
    /// # Behavior
    ///
    /// The cursor is moved to the specified position, allowing subsequent writes to
    /// overwrite bits at that position. This method does not expand the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_byte(0xef);
    /// writer.write_byte(0xFF);
    /// writer.set_cursor_at_in_bit(8 + 3);
    /// writer.write_bit(false); // Write the 3th(0-origin) bit of the byte
    /// writer.write_bit(true);  // Write the 4th(0-origin) bit of the byte
    ///
    /// assert_eq!(writer.as_slice(), &[0xef, 0b11101000]);
    ///
    /// writer.set_cursor_at_in_bit(1);
    /// assert_eq!(writer.as_slice(), &[0b10000000]);
    /// writer.set_cursor_at_in_bit(0);
    /// assert_eq!(writer.as_slice(), &[]);
    /// ```
    fn set_cursor_at_in_bit(&mut self, bit_cursor: usize) {
        let byte_pos = bit_cursor.div_ceil(8);
        self.bit_pos = bit_cursor % 8;
        if self.bit_pos == 0 {
            self.bit_pos = 8;
        }

        if byte_pos <= self.buffer.len() {
            self.buffer.truncate(byte_pos);
            if self.bit_pos != 8 {
                // Clear the bits after the new cursor position in the last byte
                let mask = 0xFF << (8 - self.bit_pos);
                let last_byte = self.buffer.last_mut().unwrap();
                *last_byte &= mask;
            }
        } else {
            self.buffer.resize(byte_pos, 0);
        }
    }

    /// Returns the number of bits written to the buffer.
    ///
    /// # Returns
    ///
    /// - The total number of bits that have been written to the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bits(0b1011, 4);
    /// assert_eq!(writer.number_of_bits_written(), 4);
    /// ```
    fn number_of_bits_written(&self) -> usize {
        self.buffer.len() * 8 - (8 - self.bit_pos)
    }

    /// Returns a slice of the underlying byte buffer.
    ///
    /// # Returns
    ///
    /// - A slice of the byte buffer containing the written bits.
    ///
    /// # Behavior
    ///
    /// The slice may include partially written bytes. For example, if only 6 bits
    /// have been written, the last byte in the slice will contain 2 unused bits.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bits(0b1011, 4);
    /// let slice = writer.as_slice();
    ///
    /// assert_eq!(slice, &[0b10110000]);
    /// ```
    fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    /// Resets the writer, clearing the buffer and setting the cursor to the beginning.
    ///
    /// # Behavior
    ///
    /// This method allows the writer to be reused without creating a new instance.
    /// All previously written data is discarded.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// use ebi::io::bit_write::BitWrite as _;
    ///
    /// let mut writer = BufferedBitWriter::new();
    /// writer.write_bits(0b1011, 4);
    /// writer.reset();
    /// assert_eq!(writer.number_of_bits_written(), 0);
    /// ```
    fn reset(&mut self) {
        self.buffer.clear();
        self.bit_pos = 8;
    }
}
