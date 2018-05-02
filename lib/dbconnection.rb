require 'pg'

require 'pry'

class DBConnection
  attr_accessor :connect
  def initialize
    @connect = PG.connect(dbname: 'text_analyzer_dev')
  end

  def upsert(category, token)
    sql = <<~SQL
    WITH upsert AS
    (UPDATE tokens SET frequency = frequency + 1 WHERE phrase = $2 RETURNING *)
    INSERT INTO tokens (phrase, frequency, category_id)
    SELECT $2, 1, (SELECT id FROM category WHERE name = $1)
    WHERE NOT EXISTS (SELECT * FROM upsert);
    SQL
    connect.exec_params(sql, [category, token])
  end

  def remove_from_category(category, token)
    sql = <<~SQL
    UPDATE tokens SET frequency = frequency - 1
    WHERE phrase = $2
    AND category_id = (SELECT id FROM category WHERE name = $1);
    SQL
    connect.exec_params(sql, [category, token])
  end

  def delete_from_category(category, token)
    sql = <<~SQL
    DELETE FROM tokens WHERE phrase = $2
    AND category_id = (SELECT id FROM category WHERE name = $1);
    SQL
    connect.exec_params(sql, [category, token])
  end
end

class DBToken < DBConnection
  def keys
    sql = "SELECT phrase FROM tokens;"
    result = connect.exec(sql)
    result.field_values('phrase')
  end

  def count
    sql = "SELECT count(phrase) FROM tokens;"
    result = connect.exec(sql)
    result.values.first.first.to_i
  end

# possibly unused
  def update_frequency(phrase, frequency, category)
    sql = <<~SQL
    UPDATE tokens SET frequency = $2 WHERE phrase = $1
    AND category_id = (SELECT id FROM category WHERE name = $3);
    SQL
    result = connect.exec_params(sql, [phrase, frequency, category])
  end

  def delete(token)
    sql = "DELETE FROM tokens WHERE phrase = $1;"
    result = connect.exec_params(sql, [token])
  end
end

class DBData < DBConnection
  def [](category_name)
    sql = <<~SQL
    SELECT sum(frequency) AS total_tokens, count(phrase) AS examples
    FROM tokens
    JOIN category ON category.id = category_id
    WHERE category.name = $1;
    SQL
    result = connect.exec_params(sql, [category_name])
    total_tokens = result.values[0][0].to_i
    examples = result.values[0][1].to_i
    { tokens: create_token_hash(category_name),
      total_tokens: total_tokens,
      examples: examples }
  end

  def create_token_hash(category)
    sql = <<~SQL
    SELECT phrase, frequency
    FROM tokens
    WHERE category_id = (SELECT id FROM category WHERE name = $1);
    SQL
    token_hash = Hash.new
    result = connect.exec_params(sql, [category])
    result.map do |tuple|
      token_hash[tuple['phrase']] = tuple['frequency'].to_i
    end
    token_hash
  end

  def keys
    sql = "SELECT name FROM category;"
    result = connect.exec(sql)
    result.field_values('name')
  end

  def has_key?(value)
    sql = "SELECT name FROM category WHERE name = $1;"
    result = connect.exec_params(sql, [value])
    !result.values.empty?
  end

  def delete(category)
    tokens_sql = <<~SQL
    DELETE FROM tokens
    WHERE category_id = (SELECT id FROM category WHERE name = $1);
    SQL
    connect.exec_params(tokens_sql, [category])

    category_sql = "DELETE FROM category WHERE name = $1;"
    connect.exec_params(category_sql, [category])
  end
end