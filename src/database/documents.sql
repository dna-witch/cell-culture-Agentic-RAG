-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documents table (for document chunks)
create table if not exists documents (
    id bigserial primary key,
    url varchar not null,
    chunk_id integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536) not null,  -- Adjust the dimension as needed
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_id)  -- Add a unique constraint to prevent duplicate chunks
);

-- Create the documents index for better search performance
create index on documents using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_documents_metadata on documents using gin (metadata);

-- Create search function for documents (chunks)
create function match_docs(
    query_embedding vector(1536),  -- Adjust the dimension as needed for chosen embedding model
    match_count int default 10,
    filter jsonb DEFAULT '{}'::jsonb
) returns table (
    id bigint,
    url varchar,
    chunk_id integer,
    title varchar,
    summary varchar,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql as $$

begin
    return query
    -- select id, url, chunk_id, title, summary, content, metadata, 1 - (documents.embedding <=> query_embedding) as similarity
    select id, url, chunk_id, title, summary, content, metadata, 1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;


-- Supabase security settings
-- Enable Row Level Security (RLS) for the documents table
alter table documents enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
    on documents
    for select
    to public
    using (true);